# Retired kernel-eval bridge

The executable adapter and its deployment configuration have been removed.
Use the service-owned native tool and HTTPS listeners described in
[the native kernel-eval guide](../../docs/kernel-eval-native.md).

Historical source bytes, original alignment scripts, licences and provenance
remain in [the non-executable archive](../../docs/archives/kernel-eval-legacy-bridge/ARCHIVE-MANIFEST.json).
The archive is evidence and is not a runtime import or deployment source.

A live deployment retires its old unit and listeners only after native HTTP,
HTTPS, authentication, ownership and device admission pass. Preserve the old
experiment's records; record unit disable/mask and closed 11908/11909 listeners
as cutover evidence. New runs bind native 12907/12909 and the service-owned CA.
