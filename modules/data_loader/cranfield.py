import re

PATH_TO_CRAN_TXT = "datasets/cran/cran.all.1400"
PATH_TO_CRAN_QRY = "datasets/cran/cran.qry"
PATH_TO_CRAN_REL = "datasets/cran/cranqrel"


class CranItem(object):
    id: str
    title: str
    author: str
    bib: str
    text: str

    def __init__(self, id, title, author, bib, text):
        self.id = id
        self.title = title
        self.author = author
        self.bib = bib
        self.text = text

    @property
    def fulltext(self):
        return ' '.join([self.title, self.text, self.author, self.bib])

    def __str__(self):
        return f"{self.title}"

    __repr__ = __str__


ID_marker = re.compile(r"\.I.")


def parse_raw():
    def get_data(PATH_TO_FILE: str):
        """
        Reads the file and splits the text into entries at the ID marker '.I'.
        The first entry is empty, so it is removed.
        """
        with open(PATH_TO_FILE, 'r') as f:
            text = f.read().replace('\n', " ")
            lines = re.split(ID_marker, text)
            lines.pop(0)
        return lines

    txt_list = get_data(PATH_TO_CRAN_TXT)
    qry_list = get_data(PATH_TO_CRAN_QRY)

    chunk_start = re.compile(r"\.[ABTW]")
    items = []

    for line in txt_list:
        entries = re.split(chunk_start, line)
        id = entries[0].strip()
        title, author, bib, text = entries[1:]
        items.append(CranItem(id=id, title=title,
                              author=author, bib=bib, text=text))

    assert len(items) == 1400
    return items


def parse_raw_query():
    def get_data(PATH_TO_FILE: str):
        """
        Reads the file and splits the text into entries at the ID marker '.I'.
        The first entry is empty, so it is removed.
        """
        with open(PATH_TO_FILE, 'r') as f:
            text = f.read().replace('\n', " ")
            lines = re.split(ID_marker, text)
            lines.pop(0)
        return lines

    txt_list = get_data(PATH_TO_CRAN_TXT)
    qry_list = get_data(PATH_TO_CRAN_QRY)

    chunk_start = re.compile(r"\.[W]")
    items = []

    for line in qry_list:
        entries = re.split(chunk_start, line)
        id = entries[0].strip()
        query = entries[1]
        items.append((int(id), query))
    return items
