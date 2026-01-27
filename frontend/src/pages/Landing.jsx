
import { motion, useScroll, useTransform } from 'framer-motion';
import { Link } from 'react-router-dom';
import { TrendingUp, Zap, BarChart3, ShieldCheck, ArrowRight } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import heroImage from '../assets/hero.png';

export default function Landing() {
    const { user } = useAuth();
    const { scrollYProgress } = useScroll();
    const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
    const scale = useTransform(scrollYProgress, [0, 0.2], [1, 0.8]);

    const fadeInUp = {
        hidden: { opacity: 0, y: 60 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut" } }
    };

    const staggerContainer = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: { staggerChildren: 0.2 }
        }
    };

    return (
        <div className="min-h-screen bg-[#0B0F19] text-white overflow-x-hidden">
            {/* Navbar */}
            <nav className="fixed top-0 w-full z-50 bg-[#0B0F19]/80 backdrop-blur-md border-b border-white/10">
                <div className="container mx-auto px-6 py-4 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <div className="p-2 bg-emerald-500/20 rounded-lg">
                            <TrendingUp className="w-6 h-6 text-emerald-400" />
                        </div>
                        <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400">
                            DemandAI
                        </span>
                    </div>
                    <div className="flex gap-4">
                        {user ? (
                            <Link to="/dashboard" className="px-6 py-2 bg-emerald-500 hover:bg-emerald-600 rounded-full font-medium transition-all">
                                Go to Dashboard
                            </Link>
                        ) : (
                            <>
                                <Link to="/login" className="px-6 py-2 text-slate-300 hover:text-white transition-colors">
                                    Log In
                                </Link>
                                <Link to="/signup" className="px-6 py-2 bg-white text-[#0B0F19] hover:bg-slate-200 rounded-full font-medium transition-all">
                                    Sign Up
                                </Link>
                            </>
                        )}
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 container mx-auto px-6">
                <div className="grid lg:grid-cols-2 gap-12 items-center">
                    <motion.div
                        initial="hidden"
                        animate="visible"
                        variants={staggerContainer}
                    >
                        <motion.div variants={fadeInUp} className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-medium mb-6">
                            <Zap className="w-4 h-4" />
                            <span>Next-Gen AI Forecasting</span>
                        </motion.div>

                        <motion.h1 variants={fadeInUp} className="text-5xl lg:text-7xl font-bold leading-tight mb-6">
                            Predict Demand <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-500">
                                with Precision
                            </span>
                        </motion.h1>

                        <motion.p variants={fadeInUp} className="text-xl text-slate-400 mb-8 max-w-lg">
                            Harness the power of multimodal AI to forecast sales with unprecedented accuracy.
                            Combine historical data, market trends, and visual signals.
                        </motion.p>

                        <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row gap-4">
                            <Link to="/signup" className="px-8 py-4 bg-emerald-500 hover:bg-emerald-600 text-white rounded-full font-bold text-lg transition-all shadow-lg shadow-emerald-500/25 flex items-center justify-center gap-2 group">
                                Start Predicting
                                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                            </Link>
                            <a href="#features" className="px-8 py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-full font-bold text-lg transition-all flex items-center justify-center">
                                Learn More
                            </a>
                        </motion.div>
                    </motion.div>

                    <motion.div
                        style={{ opacity, scale }}
                        initial={{ opacity: 0, x: 100 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 1, delay: 0.2 }}
                        className="relative"
                    >
                        <div className="absolute inset-0 bg-gradient-to-tr from-emerald-500/20 to-blue-500/20 blur-3xl rounded-full -z-10" />
                        <motion.img
                            src={heroImage}
                            alt="AI Visualization"
                            className="w-full h-auto drop-shadow-2xl hover:scale-105 transition-transform duration-700"
                            animate={{
                                y: [-20, 20, -20],
                                rotate: [0, 2, -2, 0]
                            }}
                            transition={{
                                duration: 6,
                                repeat: Infinity,
                                ease: "easeInOut"
                            }}
                        />
                    </motion.div>
                </div>
            </section>

            {/* Features Section */}
            <section id="features" className="py-24 bg-[#0F1422]">
                <div className="container mx-auto px-6">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="text-center mb-16"
                    >
                        <h2 className="text-3xl lg:text-4xl font-bold mb-4">Why Choose DemandAI?</h2>
                        <p className="text-slate-400 max-w-2xl mx-auto">
                            Our advanced multimodal architecture analyzes more than just numbers.
                        </p>
                    </motion.div>

                    <div className="grid md:grid-cols-3 gap-8">
                        {[
                            {
                                icon: <BarChart3 className="w-8 h-8 text-blue-400" />,
                                title: "Multimodal Analysis",
                                desc: "Combines sales history, product images, and textual descriptions for holistic predictions."
                            },
                            {
                                icon: <Zap className="w-8 h-8 text-yellow-400" />,
                                title: "Real-Time Processing",
                                desc: "Get instant forecasts as you upload data. Powered by optimized MobileNetV2 & LSTM models."
                            },
                            {
                                icon: <ShieldCheck className="w-8 h-8 text-emerald-400" />,
                                title: "Enterprise Grade",
                                desc: "Secure authentication, role-based access control, and encrypted data handling."
                            }
                        ].map((feature, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 30 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: idx * 0.2 }}
                                className="p-8 bg-[#0B0F19] border border-white/5 rounded-2xl hover:border-emerald-500/30 transition-colors group"
                            >
                                <div className="mb-6 p-4 bg-white/5 rounded-xl inline-block group-hover:scale-110 transition-transform">
                                    {feature.icon}
                                </div>
                                <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                                <p className="text-slate-400 leading-relaxed">
                                    {feature.desc}
                                </p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="py-12 border-t border-white/10 text-center text-slate-500">
                <p>© 2026 DemandAI. Transforming Forecasts with Intelligence.</p>
            </footer>
        </div>
    );
}
