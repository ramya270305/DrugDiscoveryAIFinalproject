import hashlib
import json
import os
import logging
import datetime
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for auth
AUTH_FILE = 'data/users.json'
SESSION_TIMEOUT = 3600  # 1 hour session timeout

# Initialize users file if it doesn't exist
def init_auth():
    """Initialize the authentication system"""
    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, 'w') as f:
            json.dump({
                "users": [],
                "sessions": {}
            }, f)
            
        # Create default admin user
        register_user("admin", "password123")
        
def hash_password(password):
    """Create a secure hash of the password"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    """Register a new user"""
    try:
        # Load current users
        if os.path.exists(AUTH_FILE):
            with open(AUTH_FILE, 'r') as f:
                auth_data = json.load(f)
        else:
            auth_data = {"users": [], "sessions": {}}
            
        # Check if username already exists
        if any(user['username'] == username for user in auth_data['users']):
            return False, "Username already exists"
            
        # Create new user
        user_id = str(uuid.uuid4())
        auth_data['users'].append({
            "id": user_id,
            "username": username,
            "password_hash": hash_password(password),
            "created_at": time.time()
        })
        
        # Save updated data
        with open(AUTH_FILE, 'w') as f:
            json.dump(auth_data, f)
            
        return True, "User registered successfully"
        
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        return False, f"Error: {str(e)}"

def login_user(username, password):
    """Authenticate a user and create a session"""
    try:
        # Load auth data
        if not os.path.exists(AUTH_FILE):
            init_auth()
            
        with open(AUTH_FILE, 'r') as f:
            auth_data = json.load(f)
            
        # Find user
        user = None
        for u in auth_data['users']:
            if u['username'] == username:
                user = u
                break
                
        if not user:
            return False, "Invalid username or password", None
            
        # Check password
        if user['password_hash'] != hash_password(password):
            return False, "Invalid username or password", None
            
        # Create session
        session_id = str(uuid.uuid4())
        auth_data['sessions'][session_id] = {
            "user_id": user['id'],
            "created_at": time.time(),
            "expires_at": time.time() + SESSION_TIMEOUT
        }
        
        # Save session
        with open(AUTH_FILE, 'w') as f:
            json.dump(auth_data, f)
            
        return True, "Login successful", session_id
        
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return False, f"Error: {str(e)}", None

def validate_session(session_id):
    """Check if a session is valid"""
    if not session_id:
        return False, None
        
    try:
        # Load auth data
        with open(AUTH_FILE, 'r') as f:
            auth_data = json.load(f)
            
        # Check if session exists
        if session_id not in auth_data['sessions']:
            return False, None
            
        session = auth_data['sessions'][session_id]
        
        # Check if session is expired
        if session['expires_at'] < time.time():
            # Clean up expired session
            del auth_data['sessions'][session_id]
            with open(AUTH_FILE, 'w') as f:
                json.dump(auth_data, f)
            return False, None
            
        # Get user
        user_id = session['user_id']
        user = None
        for u in auth_data['users']:
            if u['id'] == user_id:
                user = u
                break
                
        if not user:
            return False, None
            
        # Extend session
        auth_data['sessions'][session_id]['expires_at'] = time.time() + SESSION_TIMEOUT
        with open(AUTH_FILE, 'w') as f:
            json.dump(auth_data, f)
            
        return True, user
        
    except Exception as e:
        logger.error(f"Error validating session: {str(e)}")
        return False, None

def logout_user(session_id):
    """End a user session"""
    if not session_id:
        return True
        
    try:
        # Load auth data
        with open(AUTH_FILE, 'r') as f:
            auth_data = json.load(f)
            
        # Remove session if it exists
        if session_id in auth_data['sessions']:
            del auth_data['sessions'][session_id]
            
        # Save updated data
        with open(AUTH_FILE, 'w') as f:
            json.dump(auth_data, f)
            
        return True
        
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        return False
        
def reset_password(username, new_password):
    """Reset a user's password"""
    try:
        # Load auth data
        if not os.path.exists(AUTH_FILE):
            return False, "User database not found"
            
        with open(AUTH_FILE, 'r') as f:
            auth_data = json.load(f)
            
        # Find user
        user_found = False
        for i, user in enumerate(auth_data['users']):
            if user['username'] == username:
                # Update password
                auth_data['users'][i]['password_hash'] = hash_password(new_password)
                user_found = True
                break
                
        if not user_found:
            return False, "Username not found"
            
        # Save updated data
        with open(AUTH_FILE, 'w') as f:
            json.dump(auth_data, f)
            
        return True, "Password reset successfully"
        
    except Exception as e:
        logger.error(f"Error resetting password: {str(e)}")
        return False, f"Error: {str(e)}"