from tree_sitter import Language, Parser
import tree_sitter_markdown as ts_md

MD_LANGUAGE = Language(ts_md.language())
parser = Parser(MD_LANGUAGE)

def split_markdown(text: bytes, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    tree = parser.parse(text)
    return [node.text.decode("utf-8") for node in tree.root_node.children]