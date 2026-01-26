import Sidebar from '../components/layout/Sidebar';
import { BarChart3 } from 'lucide-react';

export default function Analytics() {
    return (
        <div className="flex min-h-screen bg-slate-900">
            <Sidebar />
            <main className="flex-1 p-8">
                <div className="flex items-center gap-3 mb-8">
                    <BarChart3 className="w-8 h-8 text-blue-400" />
                    <h1 className="text-2xl font-bold text-white">Analytics</h1>
                </div>

                <div className="grid grid-cols-2 gap-6">
                    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                        <h3 className="text-lg font-semibold text-white mb-4">Model Performance</h3>
                        <div className="space-y-3">
                            <div className="flex justify-between">
                                <span className="text-slate-400">Accuracy</span>
                                <span className="text-emerald-400 font-medium">94%</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Predictions Made</span>
                                <span className="text-white font-medium">1,247</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Avg Response Time</span>
                                <span className="text-white font-medium">120ms</span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                        <h3 className="text-lg font-semibold text-white mb-4">Usage Statistics</h3>
                        <div className="space-y-3">
                            <div className="flex justify-between">
                                <span className="text-slate-400">Files Uploaded</span>
                                <span className="text-white font-medium">156</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Products Analyzed</span>
                                <span className="text-white font-medium">2,340</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Active Users</span>
                                <span className="text-white font-medium">24</span>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
