from cryptography.fernet import Fernet
import os


def _get_cipher():
    """获取 Fernet 加密器，密钥无效时返回 None"""
    key = os.getenv("SECRET_KEY")
    if not key or key == "你的_AES_BASE64_KEY_放在这里":
        return None
    try:
        return Fernet(key.encode())
    except Exception:
        return None


cipher = _get_cipher()


def encrypt_key(raw_key: str) -> str:
    """加密 API Key"""
    if not cipher:
        return raw_key
    return cipher.encrypt(raw_key.encode()).decode()


def decrypt_key(encrypted_key: str) -> str:
    """解密 API Key"""
    if not cipher:
        return encrypted_key
    return cipher.decrypt(encrypted_key.encode()).decode()
