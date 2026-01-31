"use client";

import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Mic, Square, Upload, FileAudio, Activity } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

export function LiveDetectionPanel() {
    const [isRecording, setIsRecording] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [prediction, setPrediction] = useState<{
        label: "REAL" | "FAKE";
        confidence: number;
        latency: number;
    } | null>(null);

    // Mock simulation of detection
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isRecording) {
            interval = setInterval(() => {
                // Randomly simulate real vs fake for demo purposes
                const isFake = Math.random() > 0.7;
                setPrediction({
                    label: isFake ? "FAKE" : "REAL",
                    confidence: 85 + Math.random() * 14,
                    latency: 110 + Math.random() * 40,
                });
            }, 2000);
        } else {
            setPrediction(null);
        }
        return () => clearInterval(interval);
    }, [isRecording]);

    const toggleRecording = () => {
        setIsRecording(!isRecording);
        setFile(null);
    };

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setIsRecording(false);
            // Simulate immediate detection for file
            setTimeout(() => {
                setPrediction({
                    label: "FAKE", // Mock result for file
                    confidence: 96.5,
                    latency: 145,
                });
            }, 1500);
        }
    };

    return (
        <section className="container mx-auto px-4 py-8">
            <Card className="w-full max-w-4xl mx-auto border-primary/20 bg-panel shadow-[0_0_50px_-12px_rgba(0,0,0,0.5)]">
                <CardHeader className="border-b border-border pb-6">
                    <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center gap-2 text-xl">
                            <Activity className="text-primary w-5 h-5" />
                            Live Detection Analysis
                        </CardTitle>
                        <div className="flex items-center gap-2">
                            <span className="flex h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                            <span className="text-xs text-gray-400 font-mono">SYSTEM READY</span>
                        </div>
                    </div>
                </CardHeader>

                <CardContent className="p-6 space-y-8">
                    {/* Controls */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-4">
                            <label className="text-sm text-gray-400 uppercase tracking-widest font-semibold">Input Source</label>
                            <div className="flex gap-4">
                                <Button
                                    onClick={toggleRecording}
                                    variant={isRecording ? "destructive" : "default"}
                                    className="w-full h-12 text-base"
                                >
                                    {isRecording ? (
                                        <>
                                            <Square className="mr-2 w-4 h-4 fill-current" /> Stop
                                        </>
                                    ) : (
                                        <>
                                            <Mic className="mr-2 w-4 h-4" /> Start Listening
                                        </>
                                    )}
                                </Button>

                                <div className="relative w-full">
                                    <input
                                        type="file"
                                        accept=".wav,.mp3"
                                        onChange={handleFileUpload}
                                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                                        disabled={isRecording}
                                    />
                                    <Button variant="secondary" className="w-full h-12 text-base">
                                        <Upload className="mr-2 w-4 h-4" /> Upload Audio
                                    </Button>
                                </div>
                            </div>
                            {file && (
                                <div className="flex items-center gap-2 text-sm text-primary bg-primary/10 p-2 rounded">
                                    <FileAudio className="w-4 h-4" />
                                    <span className="truncate">{file.name}</span>
                                </div>
                            )}
                        </div>

                        {/* Visualizer Mock */}
                        <div className="relative rounded-lg bg-black/40 border border-border h-32 flex items-center justify-center overflow-hidden">
                            {isRecording ? (
                                <div className="flex items-center gap-1 h-full w-full justify-center px-4">
                                    {[...Array(20)].map((_, i) => (
                                        <motion.div
                                            key={i}
                                            className="w-2 bg-primary/60 rounded-full"
                                            animate={{
                                                height: ["20%", "80%", "20%"],
                                            }}
                                            transition={{
                                                repeat: Infinity,
                                                duration: 0.5 + Math.random() * 0.5,
                                                delay: i * 0.05,
                                            }}
                                        />
                                    ))}
                                </div>
                            ) : (
                                <div className="text-gray-600 text-sm">Waiting for audio input...</div>
                            )}
                        </div>
                    </div>

                    {/* Detection Output */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                        <div className="space-y-2">
                            <h3 className="text-sm text-gray-400 uppercase tracking-widest font-semibold">Detection Result</h3>
                            <div className="h-24 flex items-center">
                                <AnimatePresence mode="wait">
                                    {prediction ? (
                                        <motion.div
                                            key="result"
                                            initial={{ opacity: 0, scale: 0.9 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            exit={{ opacity: 0, scale: 0.9 }}
                                            className={cn(
                                                "text-5xl font-bold tracking-tighter",
                                                prediction.label === "REAL" ? "text-success drop-shadow-[0_0_10px_rgba(0,255,157,0.5)]" : "text-danger drop-shadow-[0_0_10px_rgba(255,42,109,0.5)]"
                                            )}
                                        >
                                            {prediction.label === "REAL" ? "REAL VOICE" : "SYNTHETIC"}
                                        </motion.div>
                                    ) : (
                                        <span className="text-4xl font-bold text-gray-700">--</span>
                                    )}
                                </AnimatePresence>
                            </div>
                        </div>

                        <div className="space-y-4">
                            <div className="flex justify-between items-center pb-2 border-b border-border">
                                <span className="text-gray-400">Confidence Score</span>
                                <span className={cn(
                                    "font-mono text-xl",
                                    prediction ? "text-white" : "text-gray-600"
                                )}>{prediction ? `${prediction.confidence.toFixed(1)}%` : "--"}</span>
                            </div>
                            <div className="flex justify-between items-center pb-2 border-b border-border">
                                <span className="text-gray-400">Inference Latency</span>
                                <span className={cn(
                                    "font-mono text-xl",
                                    prediction ? "text-primary" : "text-gray-600"
                                )}>{prediction ? `${prediction.latency.toFixed(0)} ms` : "--"}</span>
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </section>
    );
}
