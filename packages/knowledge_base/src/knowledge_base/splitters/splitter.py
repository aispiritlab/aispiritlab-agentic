from tree_sitter import Language, Parser
import tree_sitter_html as ts_html
import tree_sitter_json as ts_json
import tree_sitter_markdown as ts_md

MD_LANGUAGE = Language(ts_md.language())
md_parser = Parser(MD_LANGUAGE)

HTML_LANGUAGE = Language(ts_html.language())
html_parser = Parser(HTML_LANGUAGE)

JSON_LANGUAGE = Language(ts_json.language())
json_parser = Parser(JSON_LANGUAGE)


def split_markdown(text: bytes, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    tree = md_parser.parse(text)
    return [node.text.decode("utf-8") for node in tree.root_node.children]


def split_html(text: bytes) -> list[str]:
    tree = html_parser.parse(text)
    return [node.text.decode("utf-8") for node in tree.root_node.children if node.text]


def split_json(text: bytes) -> list[str]:
    tree = json_parser.parse(text)
    return [node.text.decode("utf-8") for node in tree.root_node.children if node.text]