import unittest
from model.injuri_predection import predict_injury_likelihood

class TestInjuryPrediction(unittest.TestCase):
    def test_predict_injury_likelihood(self):
        player_info = [24, 70, 180, 1, 0.5, 4]

        expected_prediction = 'High'

        prediction = predict_injury_likelihood(player_info)

        self.assertEqual(prediction, expected_prediction)

if __name__ == '__main__':
    unittest.main()
