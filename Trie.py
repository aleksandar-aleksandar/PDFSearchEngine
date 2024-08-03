class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        current_node = self.root
        
        for character in word:
            if character not in current_node.children:
                current_node.children[character] = TrieNode()
            current_node = current_node.children[character]
            
        current_node.end_of_word = True
    
    def search(self, word: str) -> bool:
        current_node = self.root
        
        for character in word:
            if character not in current_node.children:
                return False
            current_node = current_node.children[character]
            
        return current_node.end_of_word
    
    def startsWith(self, word:str) -> bool:
        current_node = self.root
        
        for character in word:
            if character not in current_node.children:
                return False
            current_node = current_node.children[character]
            
        return True
    def findWordsWithPrefix(self, prefix: str) -> list:
        current_node = self.root
        words = []
        
        # Traverse to the end node of the prefix
        for character in prefix:
            if character not in current_node.children:
                return words  # Return empty list if prefix not found
            current_node = current_node.children[character]
        
        # Recursively collect all words from the current node
        self._findWords(current_node, prefix, words)
        return words
    
    def _findWords(self, node: TrieNode, prefix: str, words: list) -> None:
        if node.end_of_word:
            words.append(prefix)
        
        for character, child_node in node.children.items():
            self._findWords(child_node, prefix + character, words)
    def __str__(self) -> str:
        words = []
        self._findWords(self.root, '', words)
        return ', '.join(words)