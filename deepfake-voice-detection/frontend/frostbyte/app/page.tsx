"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { authService } from "@/lib/auth";
import AuthForm from "@/components/AuthForm";
import { HeroSection } from "@/components/hero-section";
import LiveCallStreamer from "@/components/LiveCallStreamer";
import { DetectionTimeline } from "@/components/detection-timeline";
import { SystemStrengths } from "@/components/system-strengths";
import Link from "next/link";

export default function Home() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showAuth, setShowAuth] = useState(false);
  const router = useRouter();

  useEffect(() => {
    if (authService.isAuthenticated()) {
      setIsAuthenticated(true);
    } else {
      setShowAuth(true);
    }
  }, []);

  const handleAuthSuccess = () => {
    setIsAuthenticated(true);
    setShowAuth(false);
    // Optionally redirect to webrtc page
    // router.push("/webrtc");
  };

  if (showAuth && !isAuthenticated) {
    return <AuthForm onAuthSuccess={handleAuthSuccess} />;
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between">
      <HeroSection />

      <div className="w-full bg-gradient-to-b from-background to-panel/50 pt-10 pb-20">
        <div id="detection-panel" className="scroll-mt-20 container mx-auto px-4">
          
          {/* Header for the section */}
          <div className="mb-10 text-center">
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
              Live Voice Analysis
            </h2>
            <p className="mt-4 text-lg text-gray-400">
              Test the system in real-time. Start the simulation and speak into your microphone.
            </p>
            {isAuthenticated && (
              <div className="mt-6">
                <Link
                  href="/webrtc"
                  className="inline-block px-6 py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-lg transition-colors"
                >
                  Go to WebRTC Calling →
                </Link>
              </div>
            )}
          </div>

          {/* THE WORKING COMPONENT IS HERE */}
          <LiveCallStreamer /> 
          
        </div>

        <DetectionTimeline />

        <div id="system-strengths" className="scroll-mt-20">
          <SystemStrengths />
        </div>
      </div>

      <footer className="w-full py-8 border-t border-border mt-auto bg-panel/30">
        <div className="container mx-auto px-4 text-center text-gray-500 text-sm">
          <p>© 2024 Echelon-FrostByte. Real-Time Deepfake Detection System.</p>
        </div>
      </footer>
    </main>
  );
}