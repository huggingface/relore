"""relore -- GitHub project memory."""

# The one place the version is written. ``pyproject.toml`` reads it from here, so the
# distribution, ``relore --version``, ``relored --version``, the ``/api/v1/status``
# payload and the wire handshake (:mod:`relore.wire`) can never disagree.
__version__ = "0.3.17"
