# VPN Setup and Troubleshooting

## Overview
The company VPN gives you secure access to internal systems (intranet, file shares, HR portal, internal dashboards) when you work outside the office network. The VPN client comes preinstalled on every company laptop. Personal devices are not allowed to connect to the VPN.

## Connecting for the first time
1. Open the VPN client from the system tray (Windows) or menu bar (macOS).
2. Enter the portal address `vpn.company.internal` if it is not already filled in.
3. Sign in with your company email and password.
4. Approve the multi-factor authentication (MFA) prompt on your phone.
5. Once the status shows "Connected", internal sites will open normally.

## Common errors and fixes
- **"Gateway unreachable"**: Your home internet is down or blocking the connection. Check that normal websites open, then restart your router. Hotel and airport Wi-Fi networks sometimes block VPN traffic; try a mobile hotspot.
- **"Authentication failed"**: Your password may have expired. Reset it through the self-service password portal, wait 15 minutes for the change to sync, then reconnect.
- **Connected but internal sites do not load**: Disconnect, clear your browser cache, and reconnect. If it still fails, run `ipconfig /flushdns` on Windows or restart the laptop.
- **VPN disconnects every few minutes**: This usually happens on weak Wi-Fi. Move closer to the router or use a wired connection. Sessions also time out automatically after 12 hours, so you will need to reconnect once a day.

## When to raise a ticket
If none of the steps above work, raise a ticket in the IT self-service portal under **Network > VPN**. Include a screenshot of the error, your location, and the time the problem started.
