"""relore -- GitHub project memory."""

#: The one place the version is written. ``pyproject.toml`` reads it from here, so the
#: distribution, ``relore --version``, ``relored --version``, the ``/api/v1/status``
#: payload and the wire handshake (:mod:`relore.wire`) can never disagree.
#:
#: The minor is the milestone the build plan considers complete; the patch is every
#: release inside it. **Bump it in the same commit as any change a client can see** --
#: a wire payload, a renderer, a CLI flag -- because the client and the daemon must run
#: the same version to talk at all (:mod:`relore.wire`), which makes this string the
#: compatibility contract rather than a label.
__version__ = "0.3.17"
