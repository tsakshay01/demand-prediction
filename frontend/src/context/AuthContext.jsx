import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

// Helper: Decode JWT and check expiry
function isTokenExpired(token) {
    if (!token) return true;
    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        const exp = payload.exp;
        // exp is in seconds, Date.now() is in milliseconds
        return exp ? Date.now() >= exp * 1000 : false;
    } catch (e) {
        return true; // Invalid token format
    }
}

export function AuthProvider({ children }) {
    const [token, setToken] = useState(() => {
        const savedToken = localStorage.getItem('authToken');
        // FIX: Check expiry on initial load
        if (savedToken && isTokenExpired(savedToken)) {
            console.warn('⚠️ Token expired on load - clearing session');
            localStorage.removeItem('authToken');
            localStorage.removeItem('userData');
            return null;
        }
        return savedToken;
    });
    const [user, setUser] = useState(() => {
        const saved = localStorage.getItem('userData');
        return saved ? JSON.parse(saved) : null;
    });

    useEffect(() => {
        if (token) {
            localStorage.setItem('authToken', token);
        } else {
            localStorage.removeItem('authToken');
        }
    }, [token]);

    useEffect(() => {
        if (user) {
            localStorage.setItem('userData', JSON.stringify(user));
        } else {
            localStorage.removeItem('userData');
        }
    }, [user]);

    // FIX: Periodically check token expiry (every 60 seconds)
    useEffect(() => {
        const interval = setInterval(() => {
            if (token && isTokenExpired(token)) {
                console.warn('⚠️ Token expired - logging out');
                logout();
            }
        }, 60000);
        return () => clearInterval(interval);
    }, [token]);

    const login = (newToken, userData = null) => {
        setToken(newToken);
        setUser(userData);
    };

    const logout = () => {
        setToken(null);
        setUser(null);
        localStorage.removeItem('csv_prediction');
        localStorage.removeItem('userData');
    };

    const isAuthenticated = !!token && !isTokenExpired(token);

    return (
        <AuthContext.Provider value={{ token, user, login, logout, isAuthenticated }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}

