"use client";

import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import { Shield, Activity } from "lucide-react";
import Link from "next/link";

export function HeroSection() {
    return (
        <section className="relative w-full min-h-[80vh] flex flex-col items-center justify-center overflow-hidden pt-20 pb-20">
            {/* Background Elements */}
            <div className="absolute inset-0 z-0">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] bg-primary/20 blur-[120px] rounded-full opacity-30 animate-pulse" />
                <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-[0.2]" />
            </div>

            <div className="container relative z-10 px-4 flex flex-col items-center text-center">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm mb-6 font-medium"
                >
                    <Shield className="w-4 h-4" />
                    <span>Professional Grade Security</span>
                </motion.div>

                <motion.h1
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                    className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight mb-6 max-w-4xl bg-clip-text text-transparent bg-gradient-to-b from-white to-white/60"
                >
                    Real-Time Deepfake Voice Authentication Detection
                </motion.h1>

                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    className="text-lg md:text-xl text-gray-400 mb-8 max-w-2xl"
                >
                    Detect AI-generated voices attempting to bypass voice biometric systems during live calls with millisecond latency.
                </motion.p>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                    className="flex flex-col sm:flex-row gap-4"
                >
                    <Button size="lg" className="text-lg h-14 px-8">
                        <Activity className="mr-2 w-5 h-5" />
                        Start Live Detection
                    </Button>
                    <Link href="/architecture">
                        <Button variant="outline" size="lg" className="text-lg h-14 px-8">
                            View System Architecture
                        </Button>
                    </Link>
                </motion.div>
            </div>

            {/* Decorative Waveform */}
            <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-background to-transparent z-10" />
        </section>
    );
}
