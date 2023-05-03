import os
from re import search, split
from PIL import Image
CWD_DIR = os.getcwd()

def natural_key(text_):
    """
    Human sort
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    atoi = lambda t: int(t) if t.isdigit() else t
    return [atoi(c) for c in split(r"(\d+)", text_)]



def save_gif(directory_, name_ = "QueueGiph") -> None:
    """Render giph from jpg files in given directory."""
    frames = [os.path.join(directory_, image) for image in os.listdir(directory_) if search('.*\.jpg', image)]
    frames.sort(key = natural_key)
    frames = [Image.open(image) for image in frames]
    giph_ = frames[0]
    while os.path.exists(os.path.join(CWD_DIR, name_ + ".gif")):
        choice = ""
        while choice not in ("Y", "N"):
            print("Warning: file {}.gif exists in current working directory. Would you like to replace it?[y/n]".format(name_))
            choice = input(">")
            choice = choice.upper()
        if choice == "N":
            print("Enter different name for file")
            name_ = input(">")
        else:
            break
    giph_.save(name_ + ".gif", format = "GIF", append_images = frames, save_all = True, duration = 1000, loop = 0)
