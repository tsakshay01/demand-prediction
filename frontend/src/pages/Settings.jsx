import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import Sidebar from '../components/layout/Sidebar';
import { Settings as SettingsIcon, User, Bell, Lock, Save, Loader2, CheckCircle, XCircle, Eye, EyeOff } from 'lucide-react';

export default function Settings() {
    const { user, token } = useAuth();
    const [notifications, setNotifications] = useState(true);

    // Password change state
    const [showPasswordForm, setShowPasswordForm] = useState(false);
    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showCurrentPwd, setShowCurrentPwd] = useState(false);
    const [showNewPwd, setShowNewPwd] = useState(false);
    const [passwordLoading, setPasswordLoading] = useState(false);
    const [passwordMessage, setPasswordMessage] = useState({ type: '', text: '' });

    // Settings save state
    const [saveLoading, setSaveLoading] = useState(false);
    const [saveMessage, setSaveMessage] = useState('');

    const handleChangePassword = async (e) => {
        e.preventDefault();
        setPasswordMessage({ type: '', text: '' });

        if (newPassword !== confirmPassword) {
            setPasswordMessage({ type: 'error', text: 'New passwords do not match' });
            return;
        }

        if (newPassword.length < 6) {
            setPasswordMessage({ type: 'error', text: 'Password must be at least 6 characters' });
            return;
        }

        setPasswordLoading(true);

        try {
            const res = await fetch('/api/user/change-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ currentPassword, newPassword })
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'Failed to change password');
            }

            setPasswordMessage({ type: 'success', text: 'Password changed successfully!' });
            setCurrentPassword('');
            setNewPassword('');
            setConfirmPassword('');
            setShowPasswordForm(false);
        } catch (err) {
            setPasswordMessage({ type: 'error', text: err.message });
        } finally {
            setPasswordLoading(false);
        }
    };

    const handleSaveSettings = async () => {
        setSaveLoading(true);
        try {
            const res = await fetch('/api/user/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ notifications })
            });

            if (res.ok) {
                setSaveMessage('Settings saved!');
                setTimeout(() => setSaveMessage(''), 3000);
            }
        } catch (err) {
            console.error('Failed to save settings:', err);
        } finally {
            setSaveLoading(false);
        }
    };

    return (
        <div className="flex min-h-screen bg-slate-900">
            <Sidebar />
            <main className="flex-1 p-8">
                <div className="flex items-center gap-3 mb-8">
                    <SettingsIcon className="w-8 h-8 text-slate-400" />
                    <h1 className="text-2xl font-bold text-white">Settings</h1>
                </div>

                <div className="max-w-2xl space-y-6">
                    {/* Profile Section */}
                    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <User className="w-5 h-5 text-blue-400" />
                            <h3 className="text-lg font-semibold text-white">Profile</h3>
                        </div>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-1">Email</label>
                                <input
                                    type="email"
                                    value={user?.email || ''}
                                    disabled
                                    className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-slate-400"
                                />
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-1">Role</label>
                                <input
                                    type="text"
                                    value={user?.role === 'admin' ? 'Administrator' : 'User'}
                                    disabled
                                    className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-slate-400"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Notifications Section */}
                    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <Bell className="w-5 h-5 text-amber-400" />
                            <h3 className="text-lg font-semibold text-white">Notifications</h3>
                        </div>
                        <div className="flex items-center justify-between">
                            <span className="text-slate-300">Email Notifications</span>
                            <button
                                onClick={() => setNotifications(!notifications)}
                                className={`w-12 h-6 rounded-full transition-colors ${notifications ? 'bg-emerald-500' : 'bg-slate-600'}`}
                            >
                                <div className={`w-5 h-5 bg-white rounded-full transform transition-transform ${notifications ? 'translate-x-6' : 'translate-x-1'}`}></div>
                            </button>
                        </div>
                    </div>

                    {/* Security Section */}
                    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <Lock className="w-5 h-5 text-purple-400" />
                            <h3 className="text-lg font-semibold text-white">Security</h3>
                        </div>

                        {/* Password change message */}
                        {passwordMessage.text && (
                            <div className={`flex items-center gap-2 p-3 rounded-lg mb-4 ${passwordMessage.type === 'success'
                                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                    : 'bg-red-500/20 text-red-400 border border-red-500/30'
                                }`}>
                                {passwordMessage.type === 'success' ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                                {passwordMessage.text}
                            </div>
                        )}

                        {!showPasswordForm ? (
                            <button
                                onClick={() => setShowPasswordForm(true)}
                                className="text-emerald-400 hover:text-emerald-300 text-sm font-medium"
                            >
                                Change Password
                            </button>
                        ) : (
                            <form onSubmit={handleChangePassword} className="space-y-4">
                                <div>
                                    <label className="block text-sm text-slate-400 mb-1">Current Password</label>
                                    <div className="relative">
                                        <input
                                            type={showCurrentPwd ? 'text' : 'password'}
                                            value={currentPassword}
                                            onChange={(e) => setCurrentPassword(e.target.value)}
                                            required
                                            className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-white pr-10"
                                        />
                                        <button
                                            type="button"
                                            onClick={() => setShowCurrentPwd(!showCurrentPwd)}
                                            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                                        >
                                            {showCurrentPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                        </button>
                                    </div>
                                </div>
                                <div>
                                    <label className="block text-sm text-slate-400 mb-1">New Password</label>
                                    <div className="relative">
                                        <input
                                            type={showNewPwd ? 'text' : 'password'}
                                            value={newPassword}
                                            onChange={(e) => setNewPassword(e.target.value)}
                                            required
                                            minLength={6}
                                            className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-white pr-10"
                                        />
                                        <button
                                            type="button"
                                            onClick={() => setShowNewPwd(!showNewPwd)}
                                            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                                        >
                                            {showNewPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                        </button>
                                    </div>
                                </div>
                                <div>
                                    <label className="block text-sm text-slate-400 mb-1">Confirm New Password</label>
                                    <input
                                        type="password"
                                        value={confirmPassword}
                                        onChange={(e) => setConfirmPassword(e.target.value)}
                                        required
                                        className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-white"
                                    />
                                </div>
                                <div className="flex gap-3">
                                    <button
                                        type="submit"
                                        disabled={passwordLoading}
                                        className="flex items-center gap-2 bg-emerald-500 hover:bg-emerald-600 disabled:bg-emerald-500/50 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                                    >
                                        {passwordLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Lock className="w-4 h-4" />}
                                        Update Password
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => {
                                            setShowPasswordForm(false);
                                            setCurrentPassword('');
                                            setNewPassword('');
                                            setConfirmPassword('');
                                            setPasswordMessage({ type: '', text: '' });
                                        }}
                                        className="text-slate-400 hover:text-white px-4 py-2 text-sm"
                                    >
                                        Cancel
                                    </button>
                                </div>
                            </form>
                        )}
                    </div>

                    {/* Save Button */}
                    <div className="flex items-center gap-4">
                        <button
                            onClick={handleSaveSettings}
                            disabled={saveLoading}
                            className="flex items-center gap-2 bg-emerald-500 hover:bg-emerald-600 disabled:bg-emerald-500/50 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                        >
                            {saveLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
                            Save Changes
                        </button>
                        {saveMessage && (
                            <span className="text-emerald-400 flex items-center gap-2">
                                <CheckCircle className="w-4 h-4" />
                                {saveMessage}
                            </span>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
