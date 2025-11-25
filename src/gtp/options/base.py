import dataclasses
from dataclasses import dataclass
from typing import get_args


import click


@dataclass
class BaseOptions:
    @classmethod
    def click_options(cls):
        fields = dataclasses.fields(cls)

        def wrapper(function):
            for f in fields:
                kwargs = {"default": f.default}

                all_types = get_args(f.type)
                if len(all_types) > 0:
                    kwargs["type"] = str  # TODO handle union types somehow
                else:
                    kwargs["type"] = f.type
                if f.type is bool:
                    kwargs["is_flag"] = True

                function = click.option(f"--{f.name}", **kwargs)(function)
            return function

        return wrapper
