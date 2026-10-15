# Laptop Encryption and Recovery Keys

## Why laptops are encrypted
All company laptops use full-disk encryption (BitLocker on Windows, FileVault on macOS). If a laptop is lost or stolen, the data on it cannot be read without your credentials. Encryption is turned on automatically during setup and must never be disabled.

## BitLocker recovery screen
Sometimes Windows shows a blue BitLocker recovery screen asking for a 48-digit recovery key. This usually happens after a BIOS or firmware update, a hardware change such as a new docking station, or several failed sign-in attempts.

To get your recovery key:
1. On another device (such as your phone), go to `myaccount.company.internal` and sign in.
2. Open **Devices**, choose your laptop, and select **View BitLocker keys**.
3. Match the Key ID shown on the recovery screen and type in the 48-digit key.

If you cannot sign in on another device, call the IT Service Desk. After verifying your identity, the agent can read the recovery key to you.

## FileVault on macOS
If your Mac asks for a FileVault recovery key, contact the IT Service Desk. Mac recovery keys are stored centrally and are only given out after identity verification.

## Important rules
- Never write your recovery key on the laptop or store it in the laptop bag.
- Recovery keys change after each use for security.
- Do not try to decrypt or reformat the laptop yourself.
