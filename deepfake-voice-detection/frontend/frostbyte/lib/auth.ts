// Simplified authentication - no passwords, just username

export interface User {
  userId: number;
  username: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

export const authService = {
  async register(username: string): Promise<User> {
    const response = await fetch(`${API_URL}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Registration failed' }));
      throw new Error(errorData.error || 'Registration failed');
    }

    const data = await response.json();
    const user = { userId: data.userId, username: data.username };
    localStorage.setItem('user', JSON.stringify(user));
    return user;
  },

  async login(userIdOrUsername: string): Promise<User> {
    // Try as userId first, then as username
    const userId = parseInt(userIdOrUsername);
    const isUserId = !isNaN(userId);
    
    const response = await fetch(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        isUserId ? { userId } : { username: userIdOrUsername }
      ),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Login failed' }));
      throw new Error(errorData.error || 'Login failed');
    }

    const data = await response.json();
    const user = { userId: data.userId, username: data.username };
    localStorage.setItem('user', JSON.stringify(user));
    return user;
  },

  logout(): void {
    localStorage.removeItem('user');
  },

  getUser(): User | null {
    const userStr = localStorage.getItem('user');
    return userStr ? JSON.parse(userStr) : null;
  },

  isAuthenticated(): boolean {
    return !!this.getUser();
  },
};
