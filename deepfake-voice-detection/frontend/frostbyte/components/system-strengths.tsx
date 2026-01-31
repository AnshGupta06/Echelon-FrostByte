import { FeatureCard } from "./feature-card";
import { Clock, Phone, Brain, Zap, Activity, BarChart3 } from "lucide-react";

export function SystemStrengths() {
    const features = [
        {
            icon: Clock,
            title: "Short Audio Segment",
            description: "Accurate detection on as little as 3-10 seconds of audio input."
        },
        {
            icon: Phone,
            title: "VoIP & Codec Robust",
            description: "Resilient inference even with compression artifacts from Zoom/Teams/GSM."
        },
        {
            icon: Brain,
            title: "Neural Artifact Detection",
            description: "Specialized CNNs trained to spot vocoder inconsistencies unseen by human ears."
        },
        {
            icon: Zap,
            title: "Low-Latency Inference",
            description: "Real-time processing with <150ms latency for seamless call integration."
        },
        {
            icon: Activity,
            title: "Deepfake vs Stress",
            description: "Discriminates between natural voice stress and synthetic generation artifacts."
        },
        {
            icon: BarChart3,
            title: "Real-time Risk Scoring",
            description: "Continuous rolling window analysis provides an evolving trust score."
        }
    ];

    return (
        <section className="container mx-auto px-4 py-16">
            <div className="text-center mb-12">
                <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400 mb-4">
                    System Capabilities
                </h2>
                <p className="text-gray-400 max-w-2xl mx-auto">
                    Our multi-modal architecture combines signal processing with deep learning to deliver state-of-the-art protection.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {features.map((feature, idx) => (
                    <FeatureCard
                        key={idx}
                        icon={feature.icon}
                        title={feature.title}
                        description={feature.description}
                    />
                ))}
            </div>
        </section>
    );
}
