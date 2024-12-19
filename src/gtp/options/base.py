import dataclasses
from dataclasses import dataclass

import click


@dataclass
class BaseOptions:
    @classmethod
    def click_options(cls):
        fields = dataclasses.fields(cls)

        def wrapper(function):
            for f in fields:
                kwargs = {"type": f.type, "default": f.default}
                if f.type is bool:
                    kwargs["is_flag"] = True
                function = click.option(f"--{f.name}", **kwargs)(function)
            return function

        return wrapper
