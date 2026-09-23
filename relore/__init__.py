"""relore -- GitHub project memory."""

#: The one place the version is written. ``pyproject.toml`` reads it from here, so the
#: distribution, ``relore --version``, ``relored --version``, the ``/api/v1/status``
#: payload and the wire handshake (:mod:`relore.wire`) can never disagree.
#:
#: The minor is the milestone the build plan considers complete; the patch is every
#: release inside it. It matters because the client and the daemon must run the same
#: version to talk at all (:mod:`relore.wire`), which makes this string the compatibility
#: contract rather than a label.
#:
#: **A pull request does not touch it.** The bump belongs to the release, which is the
#: only moment anyone can say what shipped -- a bump per change makes this line the one
#: conflict every branch gets, on the one line where resolving it wrongly is invisible.
__version__ = "0.3.17"
