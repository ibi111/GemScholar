from scholarly import scholarly

class ProfessorSearch:
    def __init__(self, author_name, university):
        self.author_name = author_name
        self.university = university
        self.author_full = None
        self.author_publications = None

    def search_author(self):
        self.author_full = scholarly.fill(next(scholarly.search_author(self.author_name +','+ self.university)) , ['basics', 'indices', 'counts','publications' ,'public_access'])
        return self.author_full


if __name__ == '__main__':
    author_name = 'Robert Tibshirani'
    university = 'Stanford University'
    author_search = ProfessorSearch(author_name, university)
    author_full = author_search.search_author()
    print(author_full)