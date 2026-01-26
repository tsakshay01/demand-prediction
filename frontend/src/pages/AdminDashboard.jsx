import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
    ShieldCheck, LayoutDashboard, UploadCloud, ExternalLink,
    BarChart2, LogOut, RefreshCw, Zap, Users, Server, Activity,
    Trash2, FileText, User
} from 'lucide-react';

export default function AdminDashboard() {
    const { token, user, logout } = useAuth();
    const navigate = useNavigate();
    const [users, setUsers] = useState([]);
    const [files, setFiles] = useState([]);
    const [stats, setStats] = useState({ totalUsers: 0, activeModels: 1, systemHealth: 'Healthy' });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadAdminData();
    }, []);

    const loadAdminData = async () => {
        setLoading(true);
        try {
            // Fetch users
            const usersRes = await fetch('/api/admin/users', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (usersRes.ok) {
                const usersData = await usersRes.json();
                setUsers(usersData);
                setStats(prev => ({ ...prev, totalUsers: usersData.length }));
            }

            // Fetch files
            const filesRes = await fetch('/api/admin/files', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (filesRes.ok) {
                const filesData = await filesRes.json();
                setFiles(filesData);
            }
        } catch (err) {
            console.error('Failed to load admin data:', err);
        }
        setLoading(false);
    };

    const handleDeleteUser = async (userId) => {
        if (!confirm('Are you sure you want to delete this user?')) return;
        try {
            const res = await fetch(`/api/admin/users/${userId}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` }
            });

            const data = await res.json();

            if (res.ok) {
                setUsers(users.filter(u => u.id !== userId));
                setStats(prev => ({ ...prev, totalUsers: prev.totalUsers - 1 }));
            } else {
                alert(data.message || data.error || 'Failed to delete user');
            }
        } catch (err) {
            console.error('Failed to delete user:', err);
            alert('Failed to delete user. Check console for details.');
        }
    };

    const handleRetrain = async () => {
        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            const data = await res.json();
            alert(data.message || 'Training initiated');
        } catch (err) {
            alert('Failed to trigger training');
        }
    };

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    return (
        <div className="flex min-h-screen bg-slate-900 text-white">
            {/* Sidebar */}
            <aside className="w-64 bg-slate-950 border-r border-slate-800 flex flex-col fixed h-full">
                <div className="h-16 flex items-center px-6 border-b border-slate-800">
                    <div className="flex items-center gap-2 text-rose-500 font-bold text-xl">
                        <ShieldCheck className="w-7 h-7" />
                        <span>AdminPortal</span>
                    </div>
                </div>
                <nav className="flex-1 py-6 space-y-1">
                    <Link to="/admin" className="flex items-center gap-3 px-6 py-3 text-slate-100 bg-slate-800 border-r-2 border-rose-500">
                        <LayoutDashboard className="w-5 h-5" />
                        <span className="font-medium">Overview</span>
                    </Link>
                    <Link to="/upload" className="flex items-center gap-3 px-6 py-3 text-slate-400 hover:text-white hover:bg-slate-900 transition">
                        <UploadCloud className="w-5 h-5" />
                        <span className="font-medium">Uploads</span>
                    </Link>
                    <Link to="/" className="flex items-center gap-3 px-6 py-3 text-slate-400 hover:text-white hover:bg-slate-900 transition">
                        <ExternalLink className="w-5 h-5" />
                        <span className="font-medium">User View</span>
                    </Link>
                </nav>
                <div className="p-4 border-t border-slate-800">
                    {/* User Display */}
                    <div className="flex items-center gap-3 px-2 py-3 mb-2">
                        <div className="p-2 bg-slate-800 rounded-full">
                            <User className="w-4 h-4 text-rose-400" />
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm text-white truncate">{user?.email || 'Admin'}</p>
                            <p className="text-xs text-rose-400">Administrator</p>
                        </div>
                    </div>
                    <button onClick={handleLogout} className="flex items-center gap-2 w-full px-2 py-2 text-slate-400 hover:text-white transition">
                        <LogOut className="w-5 h-5" />
                        <span>Sign Out</span>
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className="ml-64 flex-1 p-8">
                <header className="flex justify-between items-center mb-8">
                    <div>
                        <h1 className="text-2xl font-bold">System Overview</h1>
                        <p className="text-slate-400">Manage users and system resources</p>
                    </div>
                    <div className="px-4 py-2 bg-slate-900 rounded-lg border border-slate-800 text-sm flex items-center gap-4">
                        <span className="text-slate-400">Server Status:</span>
                        <span className="text-emerald-400 font-medium">● Online</span>
                        <span className="text-xs bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-1 rounded flex items-center gap-1">
                            <Zap className="w-3 h-3" /> TFT Model
                        </span>
                        <button onClick={handleRetrain} className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white text-xs rounded transition-colors flex items-center gap-1">
                            <RefreshCw className="w-3 h-3" /> Retrain Model
                        </button>
                    </div>
                </header>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl flex items-center gap-4">
                        <Users className="w-10 h-10 text-blue-400" />
                        <div>
                            <p className="text-slate-400 text-sm">Total Users</p>
                            <h3 className="text-3xl font-bold mt-1">{stats.totalUsers}</h3>
                        </div>
                    </div>
                    <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl flex items-center gap-4">
                        <Server className="w-10 h-10 text-purple-400" />
                        <div>
                            <p className="text-slate-400 text-sm">Active Models</p>
                            <h3 className="text-3xl font-bold mt-1 text-blue-400">{stats.activeModels}</h3>
                        </div>
                    </div>
                    <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl flex items-center gap-4">
                        <Activity className="w-10 h-10 text-emerald-400" />
                        <div>
                            <p className="text-slate-400 text-sm">System Health</p>
                            <h3 className="text-3xl font-bold mt-1 text-emerald-400">{stats.systemHealth}</h3>
                        </div>
                    </div>
                </div>

                {/* User Table */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden mb-8">
                    <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
                        <h3 className="font-semibold">Registered Users</h3>
                        <button onClick={loadAdminData} className="text-sm text-blue-400 hover:text-white flex items-center gap-1">
                            <RefreshCw className="w-3 h-3" /> Refresh
                        </button>
                    </div>
                    <table className="w-full text-left text-sm">
                        <thead className="bg-slate-950 text-slate-400">
                            <tr>
                                <th className="px-6 py-3">User</th>
                                <th className="px-6 py-3">Role</th>
                                <th className="px-6 py-3">Joined</th>
                                <th className="px-6 py-3 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                            {loading ? (
                                <tr><td colSpan="4" className="px-6 py-8 text-center text-slate-500">Loading...</td></tr>
                            ) : users.length === 0 ? (
                                <tr><td colSpan="4" className="px-6 py-8 text-center text-slate-500">No users found</td></tr>
                            ) : users.map(user => (
                                <tr key={user.id} className="hover:bg-slate-800/50">
                                    <td className="px-6 py-4">{user.email}</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2 py-1 rounded text-xs font-medium ${user.role === 'admin'
                                            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                                            : 'bg-slate-700 text-slate-300'
                                            }`}>
                                            {user.role || 'user'}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-slate-400">{user.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}</td>
                                    <td className="px-6 py-4 text-right">
                                        {user.role !== 'admin' && (
                                            <button onClick={() => handleDeleteUser(user.id)} className="text-red-400 hover:text-red-300">
                                                <Trash2 className="w-4 h-4" />
                                            </button>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Files Table */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                    <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
                        <h3 className="font-semibold">System Files</h3>
                        <span className="text-xs text-slate-500">Shared Storage</span>
                    </div>
                    <table className="w-full text-left text-sm">
                        <thead className="bg-slate-950 text-slate-400">
                            <tr>
                                <th className="px-6 py-3">File</th>
                                <th className="px-6 py-3">Owner</th>
                                <th className="px-6 py-3">Size</th>
                                <th className="px-6 py-3 text-right">Uploaded</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                            {files.length === 0 ? (
                                <tr><td colSpan="4" className="px-6 py-8 text-center text-slate-500">No files uploaded</td></tr>
                            ) : files.map((file, idx) => (
                                <tr key={idx} className="hover:bg-slate-800/50">
                                    <td className="px-6 py-4 flex items-center gap-2">
                                        <FileText className="w-4 h-4 text-blue-400" />
                                        {file.filename}
                                    </td>
                                    <td className="px-6 py-4 text-slate-400">{file.user_email || 'Unknown'}</td>
                                    <td className="px-6 py-4 text-slate-400">{file.size || 'N/A'}</td>
                                    <td className="px-6 py-4 text-right text-slate-400">{file.uploaded_at ? new Date(file.uploaded_at).toLocaleDateString() : 'N/A'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </main>
        </div>
    );
}
