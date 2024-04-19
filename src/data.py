from dataclasses import dataclass

@dataclass
class Data:
    index: str
    rating: int # 1-5 for train data, -1 for test data
    title: str
    text: str
    helpful_vote: int
    verified_purchase: bool 
    processed_text: str

    def __str__(self):
        return str(self.__dict__) 