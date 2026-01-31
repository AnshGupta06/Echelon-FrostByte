"use client";

import { motion } from "framer-motion";
import { Mic, Radio, Cpu, Layers, Settings, Brain, Shield, AlertTriangle } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";

const architectureSteps = [
  {
    id: 1,
    title: "Voice Source",
    subtitle: "Live / VoIP / AI",
    icon: Mic,
    description: "Input from various voice sources including live audio, VoIP calls, or AI-generated content"
  },
  {
    id: 2,
    title: "Audio Capture Layer",
    subtitle: "Streaming",
    icon: Radio,
    description: "Real-time audio streaming and capture from the input source"
  },
  {
    id: 3,
    title: "Chunking & Buffer",
    subtitle: "3–10s windows",
    icon: Layers,
    description: "Segmentation of audio into manageable time windows for processing"
  },
  {
    id: 4,
    title: "Preprocessing",
    subtitle: "+ VoIP Simulation",
    icon: Settings,
    description: "Audio preprocessing and VoIP channel simulation for realistic testing"
  },
  {
    id: 5,
    title: "Feature Extraction",
    subtitle: "Spectrograms",
    icon: Cpu,
    description: "Extraction of spectral features and spectrogram generation for analysis"
  },
  {
    id: 6,
    title: "CNN Detection Model",
    subtitle: "Deep Learning",
    icon: Brain,
    description: "Convolutional Neural Network model for deepfake voice detection"
  },
  {
    id: 7,
    title: "Decision Engine",
    subtitle: "Temporal logic",
    icon: Shield,
    description: "Temporal analysis and decision logic for final authentication verdict"
  },
  {
    id: 8,
    title: "Output / Alert",
    subtitle: "Auth decision",
    icon: AlertTriangle,
    description: "Final authentication decision and alert generation"
  }
];

export default function ArchitecturePage() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-background to-panel/50">
      {/* Header */}
      <div className="container mx-auto px-4 pt-12 pb-8">
        <Link href="/">
          <Button variant="outline" className="mb-6">
            <ArrowLeft className="mr-2 w-4 h-4" />
            Back to Home
          </Button>
        </Link>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-4 bg-clip-text text-transparent bg-gradient-to-b from-white to-white/60">
            System Architecture
          </h1>
          <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto">
            Real-time deepfake voice detection pipeline with millisecond latency
          </p>
        </motion.div>
      </div>

      {/* Architecture Flow */}
      <div className="container mx-auto px-4 pb-20">
        <div className="max-w-4xl mx-auto">
          <div className="space-y-6">
            {architectureSteps.map((step, index) => {
              const Icon = step.icon;
              const isLast = index === architectureSteps.length - 1;
              
              return (
                <div key={step.id} className="relative">
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="relative"
                  >
                    {/* Connection Arrow */}
                    {!isLast && (
                      <div className="absolute left-1/2 -translate-x-1/2 top-full w-0.5 h-6 bg-gradient-to-b from-primary/60 to-primary/20 z-0" />
                    )}

                    {/* Step Card */}
                    <div className="relative z-10 bg-panel/80 backdrop-blur-sm border border-primary/20 rounded-lg p-6 hover:border-primary/40 transition-all duration-300 hover:shadow-lg hover:shadow-primary/10">
                      <div className="flex items-start gap-4">
                        {/* Icon */}
                        <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center">
                          <Icon className="w-6 h-6 text-primary" />
                        </div>

                        {/* Content */}
                        <div className="flex-1">
                          <div className="flex items-baseline gap-3 mb-2">
                            <h3 className="text-xl font-bold text-white">
                              {step.title}
                            </h3>
                            <span className="text-sm text-primary/80 font-mono">
                              {step.subtitle}
                            </span>
                          </div>
                          <p className="text-gray-400 text-sm">
                            {step.description}
                          </p>
                        </div>

                        {/* Step Number */}
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/20 border border-primary/40 flex items-center justify-center text-primary text-sm font-bold">
                          {step.id}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Additional Info Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
          className="mt-16 max-w-4xl mx-auto"
        >
          <div className="bg-panel/60 backdrop-blur-sm border border-primary/20 rounded-lg p-8">
            <h2 className="text-2xl font-bold text-white mb-4">Key Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0" />
                <div>
                  <h4 className="text-white font-semibold mb-1">Real-Time Processing</h4>
                  <p className="text-gray-400 text-sm">Millisecond latency for live authentication decisions</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0" />
                <div>
                  <h4 className="text-white font-semibold mb-1">VoIP Simulation</h4>
                  <p className="text-gray-400 text-sm">Realistic channel simulation for accurate testing</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0" />
                <div>
                  <h4 className="text-white font-semibold mb-1">Temporal Analysis</h4>
                  <p className="text-gray-400 text-sm">Multi-window decision logic for robust detection</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0" />
                <div>
                  <h4 className="text-white font-semibold mb-1">CNN-Based Detection</h4>
                  <p className="text-gray-400 text-sm">Deep learning model trained on spectrogram features</p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </main>
  );
}
