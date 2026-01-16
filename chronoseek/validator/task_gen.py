import random
from typing import Tuple

class SyntheticTaskGenerator:
    """
    Generates synthetic video moment retrieval tasks.
    In production, this would use a VLM (e.g., GPT-4o) to watch random clips
    and generate captions.
    """
    
    def __init__(self):
        # Placeholder dataset for testing
        # Format: (video_url, query, (start_time, end_time))
        self.dataset = [
            (
                "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
                "the big rabbit wakes up from his hole",
                (10.0, 25.0)
            ),
            (
                "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
                "two characters are arguing on the bridge",
                (50.0, 70.0)
            ),
            (
                "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
                "a robot runs down the street",
                (120.0, 135.0)
            )
        ]
    
    def generate_task(self) -> Tuple[str, str, Tuple[float, float]]:
        """
        Returns: (video_url, query, ground_truth_interval)
        """
        # Randomly select a task
        # TODO: Implement VLM pipeline here
        # 1. Fetch random CC video
        # 2. Extract random clip
        # 3. Caption with VLM
        # 4. Verify uniqueness
        return random.choice(self.dataset)
