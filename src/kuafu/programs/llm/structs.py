from collections import namedtuple


class TypedEntity(str):

    def __new__(cls, name, var_type):
        assert isinstance(var_type, Type)
        obj = str.__new__(cls, name)
        obj.name = name
        obj.var_type = var_type
        obj._str = str(obj.name) + ":" + str(obj.var_type)
        obj.is_continuous = False
        return obj

    def __str__(self):
        return self._str

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __add__(self, other):
        return str(self) + str(other)

    def __radd__(self, other):
        return str(other) + str(self)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __getnewargs_ex__(self):
        return ((self.name, self.var_type), {})


class Type(str):

    is_continuous = False

    def __call__(self, entity_name):
        return TypedEntity.__new__(TypedEntity, entity_name, self)


DEFAULT_TYPE = Type("default")
VALUE_TYPE = Type("value")
NAME_TYPE = Type("name")


class Variable(object):
    """
    A variable is a temporary object that is used in residuals
    and literals. It is not a literal itself.

    Parameters
    ----------
    name : str
    code : str
    """

    def __init__(
        self,
        name,
        code,
    ):
        self.name = name
        self.code = code

    def __call__(self):
        return self.name + " = " + self.code

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)


class Potential(object):
    """
    A Term is a factory for Literals.

    Parameters
    ----------
    name : str
    arity : int
        The number of variables in the Cost.
    var_types : [ Type ]
        The Type of each variable in the Relation.
    is_negative : bool
        Whether this Relation is negative (as in a
        negative precondition).
    is_anti : bool
        Whether this Relation is anti (as in a
        negative effect).
    """

    def __init__(
        self,
        name,
        arity,
        var_types=None,
    ):
        self.name = name
        self.arity = arity
        self.var_types = var_types

    def __call__(self, *variables):
        var_list = list(variables)
        assert len(var_list) == self.arity
        return Literal(self, var_list)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    @property
    def positive(self):
        return self.__class__(self.name, self.arity, self.var_types)


class Residual(Potential):
    """
    A Residual is a factory for Residual Literals.

    Parameters
    ----------
    name : str
    arity : int
        The number of variables in the Cost.
    var_types : [ Type ]
        The Type of each variable in the Relation.
    """

    def __init__(
        self,
        name,
        arity,
        var_types=None,
    ):
        super().__init__(name, arity, var_types)


class Predicate(Potential):
    """
    A Predicate is a factory for Predicate Literal.

    Parameters
    ----------
    name : str
    arity : int
        The number of variables in the Predicate.
    var_types : [ Type ]
        The Type of each variable in the Relation.

    """

    def __init__(
        self,
        name,
        arity,
        var_types=None,
    ):
        super().__init__(name, arity, var_types)


class Literal:
    """A literal is an instance of Residual or Predicate term.

    Parameters
    ----------
    cost : cost
    variables : [ TypedEntity or str ]
    """

    def __init__(self, term: Potential, variables: dict):
        self.term = term
        self._variables = variables

        # Apply types to untyped objects
        if self.term.var_types is not None:
            for i, (expected_type, var) in enumerate(
                zip(self.term.var_types, self._variables)
            ):
                if not hasattr(var, "var_type"):
                    # Convert strings
                    self._variables[i] = expected_type(var)

        # Cache str for repr
        self._str = str(self.term) + "(" + ",".join(map(str, self._variables)) + ")"

    def set_variables(self, variables):
        self._variables = variables
        self._update_variable_caches()

    def name(self):
        return self.term.name

    def update_variable(self, var_idx, new_value):
        self._variables[var_idx] = new_value
        self._update_variable_caches()

    def _update_variable_caches(self):
        # Recompute cache
        self._str = str(self.term) + "(" + ",".join(map(str, self._variables)) + ")"

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._str

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __gt__(self, other):
        return repr(self) > repr(other)

    def holds(self, state_literals):
        raise NotImplementedError("Goals can only be LiteralConjunctions")

    def variables(self):
        return [
            v.replace("(", "").replace(")", "").replace(",", "") for v in self._variables
        ]

    def variables_typed(self):
        return [
            str(v)
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .replace(":", " - ")
            for v in self._variables
        ]
