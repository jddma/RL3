import unittest
import numpy as np
from src.rewards import recompensa_simple,recompensa_log,RECOMPENSAS
class TestRecompensas(unittest.TestCase):
    def test_neutral(self):
        self.assertAlmostEqual(recompensa_simple(100,100),0)
        self.assertAlmostEqual(recompensa_log(100,100),0)
    def test_capital_neto(self):
        antes,despues=100,99.9
        self.assertAlmostEqual(recompensa_simple(antes,despues),-0.001)
        self.assertAlmostEqual(recompensa_log(antes,despues),np.log(.999))
    def test_equivalencia(self):
        self.assertAlmostEqual(np.log1p(recompensa_simple(100,105)),recompensa_log(100,105))
    def test_interface(self):
        self.assertEqual(set(RECOMPENSAS),{'simple','log'})
if __name__=='__main__':unittest.main()
