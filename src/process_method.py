from abc import ABC, abstractmethod
from .data import Data
from .clean import text_preprocessing_pipeline

ONLY_TTITLE_AND_TEXT_FLAG = 'only_title_and_text'
CLEAN_ONLY_TTITLE_AND_TEXT_FLAG = 'clean_only_title_and_text'
MERGE_ALL_FEATURE_TO_TEXT_FLAG = 'merge_all_feature_to_text'
CLEAN_MERGE_ALL_FEATURE_TO_TEXT_FLAG = 'clean_merge_all_feature_to_text'
ONLY_12_STAR_ONLY_TITLE_AND_TEXT_FLAG = 'only_12_star_only_title_and_text'
ONLY_45_STAR_ONLY_TITLE_AND_TEXT_FLAG = 'only_45_star_only_title_and_text'
GROUP_12_AND_45_ONLY_TITLE_AND_TEXT_FLAG = 'group_12_and_45_only_title_and_text'

class ProcessMethod(ABC):
    @abstractmethod
    def process_train(self, data: Data) -> Data:
        pass

    @abstractmethod
    def process_test(self, data: Data) -> Data:
        pass

    @abstractmethod
    def process_train_dataset(self, data: list[Data]) -> list[Data]:
        pass 

    @abstractmethod
    def process_test_dataset(self, data: list[Data]) -> list[Data]:
        pass

class OnlyTitleAndText(ProcessMethod):
    def process_train(self, data: Data) -> Data:
        data.processed_text = f'{data.title} [SEP] {data.text}'
        data.rating = data.rating - 1
        return data

    def process_test(self, data: Data) -> Data:
        data.processed_text = f'{data.title} [SEP] {data.text}'
        data.rating = data.rating - 1
        return data
    
    def process_train_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_train, data))
    
    def process_test_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_test, data))
    
class CleanOnlyTitleAndText(ProcessMethod):
    def process_train(self, data: Data) -> Data:
        title_text = text_preprocessing_pipeline(data.title)
        text_text = text_preprocessing_pipeline(data.text)
        data.processed_text = f'{title_text} [SEP] {text_text}'
        data.rating = data.rating - 1
        return data

    def process_test(self, data: Data) -> Data:
        title_text = text_preprocessing_pipeline(data.title)
        text_text = text_preprocessing_pipeline(data.text)
        data.processed_text = f'{title_text} [SEP] {text_text}'
        data.rating = data.rating - 1
        return data
    
    def process_train_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_train, data))
    
    def process_test_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_test, data))
    
class Only12StarOnlyTitleAndText(ProcessMethod):
    def get_transformed_rating(self, rating: int) -> int:
        mapping = {
            0: -1,  # test
            1: 0,
            2: 1,
            3: -1,  # ignore
            4: -1,  # ignore
            5: -1   # ignore
        }
        return mapping[rating]
    
    def process_train(self, data: Data) -> Data:
        data.processed_text = f'{data.title} [SEP] {data.text}'
        data.rating = self.get_transformed_rating(data.rating)
        return data

    def process_test(self, data: Data) -> Data:
        data.processed_text = f'{data.title} [SEP] {data.text}'
        data.rating = self.get_transformed_rating(data.rating)
        return data
    
    def process_train_dataset(self, data: list[Data]) -> list[Data]:
        return list(filter(lambda x: x.rating != -1, map(self.process_train, data)))

    def process_test_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_test, data))
    
class Only45StarOnlyTitleAndText(ProcessMethod):
    def get_transformed_rating(self, rating: int) -> int:
        mapping = {
            0: -1,  # test
            1: -1,  # ignore
            2: -1,  # ignore
            3: -1,  # ignore
            4: 0,  
            5: 1
        }
        return mapping[rating]
    
    def process_train(self, data: Data) -> Data:
        data.processed_text = f'{data.title} [SEP] {data.text}'
        data.rating = self.get_transformed_rating(data.rating)
        return data

    def process_test(self, data: Data) -> Data:
        data.processed_text = f'{data.title} [SEP] {data.text}'
        data.rating = self.get_transformed_rating(data.rating)
        return data
    
    def process_train_dataset(self, data: list[Data]) -> list[Data]:
        return list(filter(lambda x: x.rating != -1, map(self.process_train, data)))

    def process_test_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_test, data))

class Group12and45OnlyTitleAndText(ProcessMethod):
    def get_transformed_rating(self, rating: int) -> int:
        mapping = {
            0: -1,  # test
            1: 0,
            2: 0,
            3: 1,
            4: 2,  
            5: 2
        }
        return mapping[rating]
    
    def process_train(self, data: Data) -> Data:
        data.processed_text = f'{data.title} [SEP] {data.text}'
        data.rating = self.get_transformed_rating(data.rating)
        return data

    def process_test(self, data: Data) -> Data:
        data.processed_text = f'{data.title} [SEP] {data.text}'
        data.rating = self.get_transformed_rating(data.rating)
        return data
    
    def process_train_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_train, data))

    def process_test_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_test, data))

class MergeAllFeatureToText(ProcessMethod):
    def process_train(self, data: Data) -> Data:
        title_part = f'This Review Title is {data.title}'
        helpful_vote_part = f'{data.helpful_vote} people think this review is helpful'
        verified_purchase_part = 'This reviewer did purchase it' if data.verified_purchase else 'This reviewer did not purchase it'
        text_part = f'and the content is: {data.text}'
        data.processed_text = f'{title_part} {text_part}.There are other information for this review, one is {verified_purchase_part} and the other is {helpful_vote_part}.'
        data.rating = data.rating - 1
        return data

    def process_test(self, data: Data) -> Data:
        title_part = f'This Review Title is {data.title}'
        helpful_vote_part = f'{data.helpful_vote} people think this review is helpful'
        verified_purchase_part = 'This reviewer did purchase it' if data.verified_purchase else 'This reviewer did not purchase it'
        text_part = f'and the content is: {data.text}'
        data.processed_text = f'{title_part} {text_part}.There are other information for this review, one is {verified_purchase_part} and the other is {helpful_vote_part}.'
        data.rating = data.rating - 1
        return data
    
    def process_train_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_train, data))
    
    def process_test_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_test, data))
    
class CleanMergeAllFeatureToText(ProcessMethod):
    def process_train(self, data: Data) -> Data:
        title_text = text_preprocessing_pipeline(data.title)
        text_text = text_preprocessing_pipeline(data.text)
        title_part = f'this review title is: {title_text}'
        helpful_vote_part = f'{data.helpful_vote} people think this review is helpful'
        verified_purchase_part = 'this reviewer did purchase it' if data.verified_purchase else 'this reviewer did not purchase it'
        text_part = f'and the content is: {text_text}'
        data.processed_text = f'{title_part} {text_part}.there are other information for this review, one is {verified_purchase_part} and the other is {helpful_vote_part}.'
        data.rating = data.rating - 1
        return data

    def process_test(self, data: Data) -> Data:
        title_text = text_preprocessing_pipeline(data.title)
        text_text = text_preprocessing_pipeline(data.text)
        title_part = f'this review title is: {title_text}'
        helpful_vote_part = f'{data.helpful_vote} people think this review is helpful'
        verified_purchase_part = 'this reviewer did purchase it' if data.verified_purchase else 'this reviewer did not purchase it'
        text_part = f'and the content is: {text_text}'
        data.processed_text = f'{title_part} {text_part}.there are other information for this review, one is {verified_purchase_part} and the other is {helpful_vote_part}.'
        data.rating = data.rating - 1
        return data
    
    def process_train_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_train, data))
    
    def process_test_dataset(self, data: list[Data]) -> list[Data]:
        return list(map(self.process_test, data))
  
def get_processed_method(processed_method_flag: str) -> ProcessMethod:
    if processed_method_flag == ONLY_TTITLE_AND_TEXT_FLAG:
        return OnlyTitleAndText()
    if processed_method_flag == MERGE_ALL_FEATURE_TO_TEXT_FLAG:
        return MergeAllFeatureToText()
    if processed_method_flag == CLEAN_ONLY_TTITLE_AND_TEXT_FLAG:
        return CleanOnlyTitleAndText()
    if processed_method_flag == CLEAN_MERGE_ALL_FEATURE_TO_TEXT_FLAG:
        return CleanMergeAllFeatureToText()
    if processed_method_flag == ONLY_12_STAR_ONLY_TITLE_AND_TEXT_FLAG:
        return Only12StarOnlyTitleAndText()
    if processed_method_flag == ONLY_45_STAR_ONLY_TITLE_AND_TEXT_FLAG:
        return Only45StarOnlyTitleAndText()
    if processed_method_flag == GROUP_12_AND_45_ONLY_TITLE_AND_TEXT_FLAG:
        return Group12and45OnlyTitleAndText()
    raise ValueError(f'Invalid processed method flag: {processed_method_flag}')

def get_choise_flag() -> list[str]:
    return [
        ONLY_TTITLE_AND_TEXT_FLAG,
        MERGE_ALL_FEATURE_TO_TEXT_FLAG,
        CLEAN_ONLY_TTITLE_AND_TEXT_FLAG,
        CLEAN_MERGE_ALL_FEATURE_TO_TEXT_FLAG,
        ONLY_12_STAR_ONLY_TITLE_AND_TEXT_FLAG,
        ONLY_45_STAR_ONLY_TITLE_AND_TEXT_FLAG,
        GROUP_12_AND_45_ONLY_TITLE_AND_TEXT_FLAG
    ]