
"""
Purpose: Get Kaggle library list
Date created: 2020-01-14

Contributor(s):
    Mark M.
"""


# Print out what we have to work with
import pkgutil
# all_mods = [(i.name, i.ispkg) for i in pkgutil.iter_modules()]
# print('\n'.join([f"{i}" for i in all_mods]))
pkg_mods = [i.name for i in pkgutil.iter_modules() if i.ispkg]
print('\n'.join(pkg_mods))