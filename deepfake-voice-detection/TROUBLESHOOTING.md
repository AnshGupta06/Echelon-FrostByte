# Troubleshooting Guide

## Registration Failed Error

If you're seeing "Registration failed" when trying to create an account, here are the most common causes and solutions:

### 1. Database Not Set Up

**Problem**: PostgreSQL database doesn't exist or tables haven't been created.

**Solution**:
```powershell
# Check if PostgreSQL is installed and in PATH
psql --version

# If not in PATH, find your PostgreSQL installation (usually):
# C:\Program Files\PostgreSQL\15\bin\psql.exe

# Create the database
createdb deepfake_calls
# OR using full path:
"C:\Program Files\PostgreSQL\15\bin\createdb.exe" -U postgres deepfake_calls

# Run the schema
psql -d deepfake_calls -f signaling-server/database.sql
# OR using full path:
"C:\Program Files\PostgreSQL\15\bin\psql.exe" -U postgres -d deepfake_calls -f signaling-server/database.sql
```

### 2. Database Connection Failed

**Problem**: The signaling server can't connect to PostgreSQL.

**Check**:
1. Is PostgreSQL service running?
   ```powershell
   Get-Service -Name postgresql*
   ```

2. Check your `.env` file in `signaling-server/`:
   ```env
   DB_USER=postgres
   DB_HOST=localhost
   DB_NAME=deepfake_calls
   DB_PASSWORD=your_postgres_password
   DB_PORT=5432
   ```

3. Test database connection:
   ```powershell
   psql -U postgres -d deepfake_calls -c "SELECT 1;"
   ```

### 3. Signaling Server Not Running

**Problem**: The frontend can't reach the signaling server.

**Check**:
1. Is the server running on port 3001?
   ```powershell
   Test-NetConnection -ComputerName localhost -Port 3001
   ```

2. Start the server:
   ```powershell
   cd signaling-server
   npm start
   ```

3. Check for errors in the server console.

### 4. Username Already Exists

**Problem**: The username you're trying to register is already taken.

**Solution**: Choose a different username.

### 5. Network/CORS Issues

**Problem**: Browser blocking the request.

**Check**:
1. Open browser developer console (F12)
2. Check the Network tab for failed requests
3. Look for CORS errors
4. Ensure `FRONTEND_URL` in `.env` matches your frontend URL

## Quick Diagnostic Steps

1. **Check if signaling server is running**:
   ```powershell
   Test-NetConnection localhost -Port 3001
   ```

2. **Check if database exists**:
   ```powershell
   psql -U postgres -l | findstr deepfake_calls
   ```

3. **Check if tables exist**:
   ```powershell
   psql -U postgres -d deepfake_calls -c "\dt"
   ```

4. **Check server logs**:
   Look at the terminal where you started the signaling server for error messages.

## Common Error Messages

- **"Database connection failed"**: PostgreSQL not running or wrong credentials
- **"Database tables not found"**: Need to run `database.sql`
- **"Username already exists"**: Choose a different username
- **"ECONNREFUSED"**: Server not running or wrong port
- **"CORS error"**: Check `FRONTEND_URL` in `.env`

## Still Having Issues?

1. Check the signaling server console for detailed error messages
2. Check browser console (F12) for network errors
3. Verify all three services are running:
   - Signaling server (port 3001)
   - ML service (port 8000)
   - Frontend (port 3000)

