import { useState, type FormEvent } from 'react';
import type { LoginProps } from '../types/interfaces'
import { apiFetch } from "../api/client";


export const Login = ({ onLogin }: LoginProps) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      const response = await apiFetch("/api/auth/login", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        throw new Error('Invalid credentials');
      }

      const data = await response.json();
      onLogin(data.user);
    } catch (err) {
      setError('Login failed. Check your credentials');
    }
  };

  return (
    <>
      <label>
        E-Mail :
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
      </label>
      <label>
        Password :
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </label>
      <div>
        <input type="checkbox" /> Remember Me
      </div>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <div className="buttons">
        <button className="login" onClick={handleSubmit}>LOGIN</button>
        <button className="signup">SIGN UP!</button>
      </div>
    </>
  );
};









