"""Retry only unfinished owned restoration; never detach a foreign callback."""
from dataclasses import dataclass


@dataclass
class Components:
    """Factory result. release, if supplied, releases ONLY caller-transferred ownership.

    A borrowed original PEFT/model is not unloaded automatically. The factory
    must clean its own partially constructed resources if it raises before return.
    """
    native: object
    cache: object = None
    cache_sha256: str | None = None
    release: object = None


_MISSING = object()


class NativeOwner:
    """Capture original native settings before ImageResponseAdapter binds hooks."""
    def __init__(self, components):
        if not isinstance(components, Components) or (components.release is not None and not callable(components.release)):
            raise TypeError('explicit Components and optional owned release callback required')
        self.components = components
        self.closed = self.closing = self.cleanup_pending = False
        self._release_pending = components.release is not None
        self._operations = []
        self._before = []
        self._sealed = False
        try:
            root = components.native.model
            self._before = [(module, key, module.__dict__.get(key, _MISSING))
                for module in root.modules()
                for key in ('training', 'gradient_checkpointing', '_gradient_checkpointing_func')]
            self._before += [(p, 'requires_grad', p.requires_grad) for p in root.parameters()]
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                failure = BaseExceptionGroup('invalid native owner and owned cleanup failed', [primary, cleanup])
                failure.cleanup_owner = self
                raise failure
            raise

    def seal(self):
        if self._sealed:
            return
        for owner, key, original in self._before:
            installed = (owner.requires_grad if key == 'requires_grad'
                         else owner.__dict__.get(key, _MISSING))
            if installed is not original:
                self._operations.append([owner, key, original, installed])
        self._sealed = True

    def check(self):
        if self.closed or self.closing:
            raise RuntimeError('Krea2 owner is closing/closed; only cleanup retry is permitted')

    def close(self):
        if self.closed:
            return
        self.closing = True
        self.seal()  # Also handles a partial adapter-construction failure.
        errors, pending = [], []
        for owner, key, original, installed in self._operations:
            try:
                current = (owner.requires_grad if key == 'requires_grad'
                           else owner.__dict__.get(key, _MISSING))
                if current is not original:
                    if current is not installed:
                        raise RuntimeError('foreign native setting retained: ' + key)
                    if key == 'requires_grad':
                        owner.requires_grad_(original)
                    elif original is _MISSING:
                        delattr(owner, key)
                    else:
                        setattr(owner, key, original)
            except BaseException as error:
                pending.append([owner, key, original, installed])
                errors.append(error)
        self._operations = pending
        # Release only after every borrowed callback/mode/gradient was restored.
        if not errors and self._release_pending:
            try:
                self.components.release()
                self._release_pending = False
            except BaseException as error:
                errors.append(error)
        self.cleanup_pending = bool(errors)
        if errors:
            # The exception retains this owner even if construction never returned.
            failure = BaseExceptionGroup('Krea2 owned cleanup pending; retry cleanup_owner.close()', errors)
            failure.cleanup_owner = self
            raise failure
        self.closed, self.closing = True, False
