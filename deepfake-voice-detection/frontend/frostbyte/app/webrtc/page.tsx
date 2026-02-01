"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { authService } from "@/lib/auth";
import WebRTCCall from "@/components/WebRTCCall";
import { LogOut, Home } from "lucide-react";
import Link from "next/link";

export default function WebRTCPage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    if (authService.isAuthenticated()) {
      setIsAuthenticated(true);
    } else {
      router.push("/");
    }
    setLoading(false);
  }, [router]);

  const handleLogout = () => {
    authService.logout();
    router.push("/");
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  const user = authService.getUser();

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 p-4">
      <div className="container mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white">Voice Call System</h1>
            <p className="text-gray-400 mt-1">
              Secure real-time voice calling with deepfake detection
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-white font-medium">{user?.username}</div>
              <div className="text-gray-400 text-sm">ID: {user?.userId}</div>
            </div>
            <Link
              href="/"
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg flex items-center gap-2"
            >
              <Home className="w-4 h-4" />
              Home
            </Link>
            <button
              onClick={handleLogout}
              className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg flex items-center gap-2"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>

        {/* Main Call Interface */}
        <WebRTCCall />
      </div>
    </main>
  );
}

