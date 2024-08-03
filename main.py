import PyPDF2
from colorama import Fore, Style, init
import re
import pickle
from Trie import Trie
from Graph import Graph
import hashlib
import PyPDF2
import fitz

init(autoreset=True)

num_of_pages = 0
trie = Trie()
graph = Graph()

filepath = 'ASP Drugi projekat.pdf'
filepath1 = 'Data Structures and Algorithms in Python.pdf'
pdf_file = filepath1

import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import magenta

def highlight_keywords_in_pdf(keywords, input_pdf_path="search_results.pdf", output_pdf_path='highlighted_results.pdf'):
    search_keywords = []
    if isinstance(keywords, str):
        search_keywords.append(keywords)
    else:
        search_keywords = keywords
    try:
        doc = fitz.open(input_pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            for keyword in search_keywords:
                occurrences = page.search_for(keyword)
                for rect in occurrences:
                    highlight = page.add_highlight_annot(rect)
                    highlight.update()

        doc.save(output_pdf_path)
        doc.close()

        print(f"Highlighted PDF saved as '{output_pdf_path}'")

    except Exception as e:
        print(f"Error: {e}")
def save_search_results_as_pdf(results, original_pdf_path, output_pdf_path='search_results.pdf', pages_to_extract=10):
    # Sort results by rank (assuming result is a tuple with [rank, page_number])
    results.sort(reverse=True, key=lambda x: x[0])
    
    with open(original_pdf_path, 'rb') as original_file:
        reader = PyPDF2.PdfReader(original_file)
        writer = PyPDF2.PdfWriter()

        for result in results[:pages_to_extract]:
            page_num = result[1]
            writer.add_page(reader.pages[page_num-1])  

        with open(output_pdf_path, 'wb') as output_file:
            writer.write(output_file)

    print(f"Search results saved as '{output_pdf_path}'.")

def compute_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def save_objects(trie, graph, file_hash, trie_file='trie.pkl', graph_file='graph.pkl', hash_file='file_hash.pkl'):
    with open(trie_file, 'wb') as f:
        pickle.dump(trie, f)
    with open(graph_file, 'wb') as f:
        pickle.dump(graph, f)
    with open(hash_file, 'wb') as f:
        pickle.dump(file_hash, f)

def load_objects(trie_file='trie.pkl', graph_file='graph.pkl', hash_file='file_hash.pkl'):
    try:
        with open(trie_file, 'rb') as f:
            trie = pickle.load(f)
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
        with open(hash_file, 'rb') as f:
            file_hash = pickle.load(f)
        return trie, graph, file_hash
    except FileNotFoundError:
        return None, None, None
    
def create_trie_and_graph(extracted_text):
    trie = Trie()
    graph = Graph()
    
    for page_number, page in enumerate(extracted_text):
        pattern1 = r'\bpage\s([1-9][0-9]{0,3}|10000)\b'
        pattern2 = r'\bsee\spages\s([1-9][0-9]{0,3}|1000)\sand\s([1-9][0-9]{0,3}|1000)\b'
        
        matches1 = re.findall(pattern1, page, re.IGNORECASE)
        matches2 = re.findall(pattern2, page, re.IGNORECASE)
        
        page = re.sub('\n', " ", page)
        graph.add_node(page_number)
        
        if matches1:
            for match in matches1:
                graph.add_edge(page_number, int(match))
        
        if matches2:
            for match in matches2:
                page_from = int(match[0])
                page_to = int(match[1])
                
                if page_from != page_to:
                    for page_num in range(page_from, page_to + 1):
                        graph.add_edge(page_number, page_num)
        
        for word in page.split():
            word = re.sub(r'[^A-Za-z0-9]+', '', word)
            word = word.strip()
            trie.insert(word)
    
    return trie, graph



def extract_text_from_pdf(pdf_file: str) -> list:
    # Open the PDF file of your choice
    with open(pdf_file, 'rb') as pdf:
        reader = PyPDF2.PdfReader(pdf)
        pdf_text = []

        for page in reader.pages:
            content = page.extract_text()
            content = re.sub(r'[^a-zA-Z0-9\s]', ' ', content)
            content = content.replace('\n','')
            pdf_text.append(content)

        return pdf_text

def handle_phrase_query(queries):
    global trie
    queries = queries.replace('"', '')
    valid = True
    for query in queries.split():
        if not trie.search(query):
            valid = False
            if trie.startsWith(query) and len(query) > 1:
                print(f"'{query}' not found")
                for sug in trie.findWordsWithPrefix(query):
                    print(f"Did you mean to look for {sug}?")
    if valid:
        results, contexts = find_word(queries)
        if results:
            rank_results(results, contexts, queries)
        else:
            print("Could not find " + queries)
    else:
        print("Could not find " + queries)

def find_most_relevant_autocomplete(prefix):
    suggestions = trie.findWordsWithPrefix(prefix)
    if not suggestions:
        return None
    
    # Example: Sort suggestions by word count in descending order
    suggestions.sort(key=lambda word: len(find_word(word)[1]), reverse=True)
    
    return suggestions[0]  # Return the most relevant suggestion

def handle_single_word_query(queries: str):
    global trie
    if "*" in queries:
        print("Autocomplete")
        queries = queries.replace("*", "")
        
        autocomplete_word = find_most_relevant_autocomplete(queries)
        if autocomplete_word:
            print("Results for", autocomplete_word)
            rank_results(find_word(autocomplete_word)[0], find_word(autocomplete_word)[1], queries)
        else:
            print("No autocomplete suggestions found for", queries)
    elif trie.search(queries):
        rank_results(find_word(queries)[0], find_word(queries)[1], queries)
    elif trie.startsWith(queries) and len(queries) > 1:
        print(f"'{queries}' not found")
        for sug in trie.findWordsWithPrefix(queries):
            print(f"Did you mean to look for {sug}?")
    else:
        print(f"'{queries}' not found")

def handle_and_query(query_list):
    global trie
    ranked_pages_mixed = {}
    repeated_pages = {}
    contexts_to_rank = []

    for _ in range(0, len(query_list)):
        if "AND" in query_list:
            query_list.remove("AND")

    for query in query_list:
        results, contexts = find_word(query)
        for context in contexts:
            contexts_to_rank.append(context)
        for result in results:
            num_of_res = result[0]
            page_num = result[1]
            if ranked_pages_mixed.get(page_num):
                ranked_pages_mixed[page_num] += num_of_res
            else:
                ranked_pages_mixed[page_num] = num_of_res
            if repeated_pages.get(page_num):
                repeated_pages[page_num] += 1
            else:
                repeated_pages[page_num] = 1

    results_to_rank = [
        [value, key] for key, value in ranked_pages_mixed.items()
        if repeated_pages[key] == len(query_list)
    ]

    rank_results(results_to_rank, contexts_to_rank, query_list)

def handle_not_query(query_list):
    global trie
    ranked_pages_mixed = {}
    not_queries = []
    contexts_to_rank = []

    for query in query_list:
        if query_list[query_list.index(query) - 1] == "NOT":
            not_queries.append(query)
            query_list.remove(query)
    for _ in range(0, len(query_list)):
        if "NOT" in query_list:
            query_list.remove("NOT")

    for query in query_list:
        results, contexts = find_word(query)
        for context in contexts:
            contexts_to_rank.append(context)
        for result in results:
            num_of_res = result[0]
            page_num = result[1]
            if ranked_pages_mixed.get(page_num):
                ranked_pages_mixed[page_num] += num_of_res
            else:
                ranked_pages_mixed[page_num] = num_of_res

    for query1 in not_queries:
        results, context = find_word(query1)
        for result in results:
            num_of_res = result[0]
            page_num = result[1]
            if num_of_res > 0:
                ranked_pages_mixed[page_num] = 0

    results_to_rank = [
        [value, key] for key, value in ranked_pages_mixed.items()
        if value > 0
    ]

    rank_results(results_to_rank, contexts_to_rank, query_list)

def handle_or_query(query_list):
    global trie
    ranked_pages_mixed = {}
    contexts_to_rank = []

    if "OR" in query_list:
        query_list.remove("OR")

    for query in query_list:
        results, contexts = find_word(query)
        for context in contexts:
            contexts_to_rank.append(context)
        for result in results:
            num_of_res = result[0]
            page_num = result[1]
            if ranked_pages_mixed.get(page_num):
                ranked_pages_mixed[page_num] += num_of_res
            else:
                ranked_pages_mixed[page_num] = num_of_res

    results_to_rank = [[value, key] for key, value in ranked_pages_mixed.items()]

    rank_results(results_to_rank, contexts_to_rank, query_list)

def find_words(queries: str):
    global trie
    num_of_queries = len(queries.split())
    # print(num_of_queries)

    if '"' in queries:
        handle_phrase_query(queries)
    elif num_of_queries == 1:
        handle_single_word_query(queries)
    else:
        query_list = queries.split()
        if "AND" in query_list:
            handle_and_query(query_list)
        elif "NOT" in query_list:
            handle_not_query(query_list)
        else:
            handle_or_query(query_list)
                        
def create_txt_files(extracted_text):
    global num_of_pages
    print(len(extracted_text), " pages loaded")
    i = 1
    for page in extracted_text:
        num_of_pages = num_of_pages + 1
        with open(f'pages/page{i}.txt', 'w', encoding='utf-16') as f:
            f.write(page)
        i += 1

def find_word(query: str):
    global num_of_pages, trie
    all_results = [] # list of all results -> [word_count, page_num]
    all_context = []
    


    query_regex = re.compile(r'\b' + re.escape(query) + r'\b', re.IGNORECASE)
    num_of_results = 1
    for i in range(1, num_of_pages + 1):
        with open(f'pages/page{i}.txt', 'r', encoding='utf-16') as f:
            content = f.read()
            word_count = len(query_regex.findall(content))

            if word_count > 0:
                all_results.append([word_count, i])
                j = 1
                '''
                print(" ")
                print(f"'{query}' in page {str(i)} is shown {word_count} times")  
                print("--------------------------------")
                '''
                for match in query_regex.finditer(content):
                    start = match.start()
                    end = match.end()
                    context = content[max(0, start-30):end+30]

                    highlighted_context = re.sub(r'\b' + re.escape(query) + r'\b',lambda match: Fore.RED + match.group(0) + Style.RESET_ALL,context,flags=re.IGNORECASE) 
                    '''
                    print("context")
                    print(f"{num_of_results}. [{i}][{j}] ..." + highlighted_context + "...")
                    '''
                    
                    all_context.append([num_of_results, i, j, highlighted_context])
                    j = j + 1
                    num_of_results = num_of_results + 1
                
    return (all_results, all_context)

def rank_results(results: list, contexts: list, queries: list):
    global graph, pdf_file
    refrences = []
    pages = []
    refrenced_pages = []
     
    number_of_results = int(input("Number of results:"))
    
    pages_to_show = len(results) - number_of_results
    
    for node in graph.adjacency_list.values():
        if node:
            for page in node:
                refrences.append(int(page))
            
    for result in results:
        pages.append(result[1])
        if result[1] in refrences:
            refrenced_pages.append(result[1])
            result[0] += 10
        
    print("Backlinks: ", refrenced_pages)
    
    sorted_result_pages = bucket_sort(results)
    # print("sorted_result_pages", sorted_result_pages)
    

    
    for result in sorted_result_pages:
        if number_of_results < 1:
            print(f"{pages_to_show} results left to show...")
            number_of_results = int(input("How many more inputs do you want to see(type 0 to leave): "))
            pages_to_show = pages_to_show - number_of_results
            if number_of_results == 0:
                break
            else:
                continue
        else:
            print("Page " + str(result[1])+ " with relevancy score of " +  str(result[0]))
            for context in contexts:
                if int(result[1]) == context[1]:
                    print("..." + context[3] + "...")
            print("--------------------------------")
            number_of_results -= 1
    
    save_pages = input("Do you want to save the first 10 results in a pdf file? (yes/no): ")
    if save_pages == "yes":
        save_search_results_as_pdf(results, pdf_file)
        highlight_keywords_in_pdf(queries)  
        
    #sort results

def bucket_sort(arr):
    if not arr:
        return arr

    # Find the minimum and maximum values
    min_value = min(arr, key=lambda x: x[0])[0]
    max_value = max(arr, key=lambda x: x[0])[0]
    
    if min_value == max_value:
        min_value = 0
        max_value = 1   

    # Number of buckets
    if len(arr) < 1:
        return arr
    else:
        bucket_count = len(arr)

    # Create empty buckets
    buckets = [[] for _ in range(bucket_count)]

    # Distribute the elements into buckets
    for item in arr:
        index = (item[0] - min_value) * (bucket_count - 1) // (max_value - min_value)
        buckets[index].append(item)
        

    # Sort each bucket and concatenate the result
    sorted_arr = []
    for bucket in reversed(buckets):
        sorted_arr.extend(sorted(bucket, key=lambda x: x[0], reverse=True))

    return sorted_arr

def find_nth(content: str, query: str, n: int) -> int:
    start = content.find(" "+query+" ")
    while start >= 0 and n > 1:
        start = content.find(query, start+len(query))
        n -= 1
    return start

def find_context(content, query, n):
    start = find_nth(content, query,n )
    n += 1
    end = start + len(query)
    context = content[max(0, start-30):min(len(content), end+30)]
    return context
          
def get_context(query:str):
    global num_of_pages
    context = []
    for i in range(1, num_of_pages + 1):
        with open(f'pages/page{i}.txt', 'r', encoding='utf-8') as f:
            content = f.read()            
            for word in content.split():
                if word.lower() == query.lower():
                    start = content.index(" "+word+" ")
                    end = start + len(word)
                    context.append(content[start-30:end+30])
    return context
                       
def main():
    global trie, graph, pdf_file

    
    current_file_hash = compute_file_hash(pdf_file)
    extracted_text = extract_text_from_pdf(pdf_file)
    trie, graph, saved_file_hash = load_objects()
    
    if trie is None or graph is None or current_file_hash != saved_file_hash:
        print("Creating new trie and graph...")
        trie, graph = create_trie_and_graph(extracted_text)
        save_objects(trie, graph, current_file_hash)
    else:
        print("Loaded trie and graph from files.")
    print(graph)
    create_txt_files(extracted_text)
    
    while True:
        # make console menu
        print("1. Search for word(s)")
        print("2. Exit")
        option = input("Choose an option:")
    
        if option == '1':
            query = input("Enter a word or more to search for: ")
            find_words(query)
        elif option == '2':
            break
        else: 
            print("Invalid option. Please try again.")

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
