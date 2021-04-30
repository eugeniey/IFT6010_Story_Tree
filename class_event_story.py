class Event:
    def __init__(self, title, content, keywords, date, vector):
        self.title = title
        self.content = content
        self.keywords = keywords
        self.date = date
        self.vector = vector

    def get_title(self):
        return self.title

    def get_content(self):
        return self.content

    def get_keywords(self):
        return self.keywords

    def get_vector(self):
        return self.vector


class Story:
    def __init__(self, event):
        self.list_of_events = [event]
        self.list_keywords = event.get_keywords()

    def add_event(self, new_event):
        self.list_of_events.append(new_event)
        # A story keywords, is the union of all keywords
        self.list_keywords.extend(new_event.get_keywords())

    def get_list_of_keywords(self):
        return self.list_keywords

    def get_list_of_events(self):
        return self.list_of_events
