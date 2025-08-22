from dataclasses import dataclass


#@dataclass
class trie_node:
    def __init__(self):
        self.children :dict[str, trie_node] ={}
        self.is_end = False
        self.word=""

    def __repr__(self, level=0):
        indent = " "*level
        lines = [f"{indent}- {self.word if self.is_end else ' '}"]
        for char, child in self.children.items():
            lines.append(f"{indent} {char}")
            lines.append(child.__repr__(level+1))
        return "\n".join(lines)

class data_trie:
    def __init__(self):
        self.root = trie_node()
        self.word_count = dict()
    
    def add_word(self, _str):
        node = self.root
        lower_case = _str.lower()

        for c in lower_case:
            if c not in node.children:
                node.children[c]=trie_node()
            node=node.children[c]
        node.is_end = True
        node.word = lower_case
        if lower_case in self.word_count:
            self.word_count[lower_case] +=1
        else:
            self.word_count[lower_case]=1

    def search(self,prefix:str, k:int):
        node = self.root
        for c in prefix.lower():
            if c not in node.children:
                return []
            node= node.children[c]
        
        results=[]
        self.dfs(node, results)
        results.sort(key=lambda w: (-self.word_count[w], w))
        
        return results[:k]

    def dfs(self, node, result):
        if node.is_end:
            result.append(node.word)
           # print("found a word")
        for child_node in node.children.values():
            self.dfs(child_node, result)

    def __repr__(self):
        return self.root.__repr__(0)


trie = data_trie()

trie.add_word("apple")
trie.add_word("app")
trie.add_word("application")
trie.add_word("app")
trie.add_word("ape")

print(trie.search("app", 2)) #→ ["app", "apple"]
print(trie.search("ap", 4)) #→ ["app", "apple", "application"]
print(trie.search("apple", 1)) # → ["apple"]






