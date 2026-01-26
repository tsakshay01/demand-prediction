import { useAuth } from '../context/AuthContext';

export function useApi() {
    const { token, logout } = useAuth();

    const apiFetch = async (url, options = {}) => {
        const headers = {
            ...options.headers,
        };

        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        // Don't set Content-Type for FormData (let browser set it with boundary)
        if (!(options.body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
        }

        const response = await fetch(url, {
            ...options,
            headers,
        });

        // Handle 401 Unauthorized
        if (response.status === 401) {
            logout();
            throw new Error('Session expired. Please login again.');
        }

        return response;
    };

    const get = (url) => apiFetch(url, { method: 'GET' });

    const post = (url, body) => {
        const options = { method: 'POST' };
        if (body instanceof FormData) {
            options.body = body;
        } else {
            options.body = JSON.stringify(body);
        }
        return apiFetch(url, options);
    };

    return { apiFetch, get, post };
}
