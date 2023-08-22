import base64

from Crypto import Random
from Crypto.Cipher import AES



class Security:

    def __init__(self):
        key = "7EBF7942_F40D_498C_8E30_5EF96792"
        self._encoding = 'utf-8'
        self.key = bytes(key, self._encoding)

    def pad(self, s):
        return s + b"\0" * (AES.block_size - len(s) % AES.block_size)


    def decrypt(self, dtext:bytes) -> str:
        """
        :param dtext: encrypted message
        :return: returns decrypted message
        """
        ciphertext = base64.b64decode(dtext)
        iv = ciphertext[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        plaintext = cipher.decrypt(ciphertext[AES.block_size:])
        return plaintext.rstrip(b"\0").decode(self._encoding)

    def string_to_bytes(self, message: str) -> bytes:
        return bytes(message, self._encoding)