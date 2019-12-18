# grams.root_exceptions
#!/usr/bin/env python3
"""Module for custom exceptions.

From *Effective Python* by Brett Slatkin,
"Having a root exception in a module makes it easy for consumers of an API to
catch all of the exceptions that are raised on purpose...preventing the API's
exceptions from propogating too far upward and breaking the calling program. It
insulates the calling code from the API. This insulation has three helpful
effects.

First, root exceptions let callers understand when there's a problem with their
usage of your API...The second advantage...is that they can help find bugs in
the API module's code...All other bugs must be the ones that weren't intended...
The third impact...is future-proofing...you could add an Exception subclass and
the calling code will continue to work as before, because it already catches
the exception being subclassed.::

    try:
        distro = grams.FreqDist([(1,2,3)])
    except root_exceptions.ImproperListFormatError:
        distro = grams.FreqDist([(1,2)])
    except root_exceptions.ImproperBinFormatError:
        distro = grams.FreqDist([])
    except root_exceptions.Error as e:
        logging.error("Bug in the calling code: %s", e)
    except Exception as e:
        logging.error("Bug in the API code: %s", e)
        raise
"""

__all__ = [
    "Error",
    "ImproperDataFormatError",
    "ImproperBinFormatError",
    "ImproperListFormatError",
    "ImproperTupleFormatError",
    "InvalidDataError",
    "InvalidTokenError",
    "InvalidTokenDatatype",
    "InvalidFrequencyError",
    "InvalidDataTypeError",
    "InvalidIndexError",
    "IndexNotFoundError",
    "KeyError",
    "KeyNotFoundError",
    "InvalidKeyError",
    "DoesNotExistError",
    "MissingDataError",
]


class Error(Exception):
    """Base-class for all exceptions raised by this module."""
    pass


class ImproperDataFormatError(Error):
    """Base-class for improperly formatted data."""
    pass


class ImproperBinFormatError(ImproperDataFormatError):
    """The provided bins were improperly formatted."""
    pass


class ImproperListFormatError(ImproperDataFormatError):
    """The provided list was improperly formatted."""
    pass


class ImproperTupleFormatError(ImproperDataFormatError):
    """The provided tuple was improperly formatted."""
    pass


class InvalidDataError(Error):
    """The provided data was not valid."""
    pass


class InvalidTokenError(InvalidDataError):
    """The provided token was not valid."""
    pass


class InvalidTokenDatatype(InvalidTokenError):
    """The provided token's datatype is invalid."""
    pass


class InvalidFrequencyError(InvalidDataError):
    """The provided frequency was not valid."""
    pass


class InvalidDataTypeError(InvalidDataError):
    """The datatype of the provided data is invalid."""
    pass


class IndexError(Error):
    """Base-class for index errors"""
    pass


class IndexNotFoundError(IndexError):
    """The value being used to index a sequence is incompatible."""
    pass


class InvalidIndexError(IndexError):
    """The value being used to index a sequence isn't a valid value."""
    pass


class KeyError(Error):
    """Base-class to the use of keys with mappable objects."""
    pass


class KeyNotFoundError(Error):
    """The provided key doesn't exist in a mappable object."""
    pass


class InvalidKeyError(KeyError):
    """The provided key can't be used to access a mappable object."""
    pass


class DoesNotExistError(Error):
    """Base-class for all the things that don't exist."""
    pass


class MissingDataError(Error):
    """There was supposed to be data, but something must've happened."""
    pass
