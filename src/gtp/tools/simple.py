def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def create_exp_info_text(species: str, wing: str, color: str, chromosome: int) -> str:
    return f"{species}_{wing}_{color}_chromosome_{chromosome}"
