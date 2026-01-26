import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import {
    TrendingUp,
    LayoutDashboard,
    Upload,
    Settings,
    LogOut,
    BarChart3,
    ShieldCheck,
    User
} from 'lucide-react';

export default function Sidebar() {
    const { logout, user } = useAuth();
    const navigate = useNavigate();
    const isAdmin = user?.role === 'admin';

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    const navItems = [
        { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
        { to: '/upload', icon: Upload, label: 'Upload' },
        { to: '/analytics', icon: BarChart3, label: 'Analytics' },
        { to: '/settings', icon: Settings, label: 'Settings' },
    ];

    // Add admin link for admin users
    if (isAdmin) {
        navItems.push({ to: '/admin', icon: ShieldCheck, label: 'Admin Portal', isAdmin: true });
    }

    return (
        <aside className="w-64 bg-slate-800/50 border-r border-slate-700 flex flex-col">
            {/* Logo */}
            <div className="p-6 border-b border-slate-700">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-emerald-500/20 rounded-lg">
                        <TrendingUp className="w-6 h-6 text-emerald-400" />
                    </div>
                    <span className="text-xl font-bold text-white">DemandAI</span>
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-4">
                <ul className="space-y-1">
                    {navItems.map((item) => (
                        <li key={item.to}>
                            <NavLink
                                to={item.to}
                                className={({ isActive }) =>
                                    `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${item.isAdmin
                                        ? isActive
                                            ? 'bg-rose-500/20 text-rose-400'
                                            : 'text-rose-400 hover:text-rose-300 hover:bg-rose-500/10'
                                        : isActive
                                            ? 'bg-emerald-500/20 text-emerald-400'
                                            : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                                    }`
                                }
                            >
                                <item.icon className="w-5 h-5" />
                                {item.label}
                            </NavLink>
                        </li>
                    ))}
                </ul>
            </nav>

            {/* User Info & Logout */}
            <div className="p-4 border-t border-slate-700">
                {/* User Display */}
                <div className="flex items-center gap-3 px-4 py-3 mb-2">
                    <div className="p-2 bg-slate-700 rounded-full">
                        <User className="w-4 h-4 text-slate-400" />
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-sm text-white truncate">{user?.email || 'User'}</p>
                        <p className={`text-xs ${isAdmin ? 'text-rose-400' : 'text-slate-500'}`}>
                            {isAdmin ? 'Administrator' : 'User'}
                        </p>
                    </div>
                </div>
                <button
                    onClick={handleLogout}
                    className="flex items-center gap-3 w-full px-4 py-3 text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors"
                >
                    <LogOut className="w-5 h-5" />
                    Sign Out
                </button>
            </div>
        </aside>
    );
}
