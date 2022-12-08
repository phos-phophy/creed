import unittest
from pathlib import Path

from src.abstract import Document, EntityFact, RelationFact, Span
from src.datasets import DocREDConverter

from .helpers import equal_docs


class DocREDConverterTest(unittest.TestCase):
    def setUp(self) -> None:
        doc_id = "Skai TV"
        text = "Skai TV is a Greek free - to - air television network based in Piraeus . It is part of the Skai Group , one of the " \
               "largest media groups in the country . It was relaunched in its present form on 1st of April 2006 in the Athens " \
               "metropolitan area , and gradually spread its coverage nationwide . Besides digital terrestrial transmission , it is " \
               "available on the subscription - based encrypted services of Nova and Cosmote TV . Skai TV is also a member of Digea , " \
               "a consortium of private television networks introducing digital terrestrial transmission in Greece . At launch , Skai " \
               "TV opted for dubbing all foreign language content into Greek , instead of using subtitles . This is very uncommon in " \
               "Greece for anything except documentaries ( using voiceover dubbing ) and children 's programmes ( using lip - synced " \
               "dubbing ) , so after intense criticism the station switched to using subtitles for almost all foreign shows ."

        sentences = [[Span(0, 4), Span(5, 7), Span(8, 10), Span(11, 12), Span(13, 18), Span(19, 23), Span(24, 25), Span(26, 28),
                      Span(29, 30), Span(31, 34), Span(35, 45), Span(46, 53), Span(54, 59), Span(60, 62), Span(63, 70), Span(71, 72)],
                     [Span(73, 75), Span(76, 78), Span(79, 83), Span(84, 86), Span(87, 90), Span(91, 95), Span(96, 101), Span(102, 103),
                      Span(104, 107), Span(108, 110), Span(111, 114), Span(115, 122), Span(123, 128), Span(129, 135), Span(136, 138),
                      Span(139, 142), Span(143, 150), Span(151, 152)],
                     [Span(153, 155), Span(156, 159), Span(160, 170), Span(171, 173), Span(174, 177), Span(178, 185), Span(186, 190),
                      Span(191, 193), Span(194, 197), Span(198, 200), Span(201, 206), Span(207, 211), Span(212, 214), Span(215, 218),
                      Span(219, 225), Span(226, 238), Span(239, 243), Span(244, 245), Span(246, 249), Span(250, 259), Span(260, 266),
                      Span(267, 270), Span(271, 279), Span(280, 290), Span(291, 292)],
                     [Span(293, 300), Span(301, 308), Span(309, 320), Span(321, 333), Span(334, 335), Span(336, 338), Span(339, 341),
                      Span(342, 351), Span(352, 354), Span(355, 358), Span(359, 371), Span(372, 373), Span(374, 379), Span(380, 389),
                      Span(390, 398), Span(399, 401), Span(402, 406), Span(407, 410), Span(411, 418), Span(419, 421), Span(422, 423)],
                     [Span(424, 428), Span(429, 431), Span(432, 434), Span(435, 439), Span(440, 441), Span(442, 448), Span(449, 451),
                      Span(452, 457), Span(458, 459), Span(460, 461), Span(462, 472), Span(473, 475), Span(476, 483), Span(484, 494),
                      Span(495, 503), Span(504, 515), Span(516, 523), Span(524, 535), Span(536, 548), Span(549, 551), Span(552, 558),
                      Span(559, 560)],
                     [Span(561, 563), Span(564, 570), Span(571, 572), Span(573, 577), Span(578, 580), Span(581, 586), Span(587, 590),
                      Span(591, 598), Span(599, 602), Span(603, 610), Span(611, 619), Span(620, 627), Span(628, 632), Span(633, 638),
                      Span(639, 640), Span(641, 648), Span(649, 651), Span(652, 657), Span(658, 667), Span(668, 669)],
                     [Span(670, 674), Span(675, 677), Span(678, 682), Span(683, 691), Span(692, 694), Span(695, 701), Span(702, 705),
                      Span(706, 714), Span(715, 721), Span(722, 735), Span(736, 737), Span(738, 743), Span(744, 753), Span(754, 761),
                      Span(762, 763), Span(764, 767), Span(768, 776), Span(777, 779), Span(780, 790), Span(791, 792), Span(793, 798),
                      Span(799, 802), Span(803, 804), Span(805, 811), Span(812, 819), Span(820, 821), Span(822, 823), Span(824, 826),
                      Span(827, 832), Span(833, 840), Span(841, 850), Span(851, 854), Span(855, 862), Span(863, 871), Span(872, 874),
                      Span(875, 880), Span(881, 890), Span(891, 894), Span(895, 901), Span(902, 905), Span(906, 913), Span(914, 919),
                      Span(920, 921)]]

        facts = [
            EntityFact("", "ORG", "0", (Span(424, 428), Span(429, 431))),
            EntityFact("", "ORG", "0", (Span(0, 4), Span(5, 7))),
            EntityFact("", "ORG", "0", (Span(573, 577), Span(578, 580))),
            EntityFact("", "LOC", "1", (Span(13, 18),)),
            EntityFact("", "LOC", "2", (Span(63, 70),)),
            EntityFact("", "ORG", "3", (Span(91, 95), Span(96, 101))),
            EntityFact("", "TIME", "4", (Span(194, 197), Span(198, 200), Span(201, 206), Span(207, 211))),
            EntityFact("", "LOC", "5", (Span(219, 225),)),
            EntityFact("", "ORG", "6", (Span(402, 406),)),
            EntityFact("", "ORG", "7", (Span(411, 418), Span(419, 421))),
            EntityFact("", "ORG", "8", (Span(452, 457),)),
            EntityFact("", "LOC", "9", (Span(552, 558),)),
            EntityFact("", "LOC", "9", (Span(695, 701),)),
            EntityFact("", "MISC", "10", (Span(633, 638),))
        ]

        facts.extend([
            RelationFact("", "P17", facts[4], facts[11]),
            RelationFact("", "P17", facts[4], facts[12]),
            RelationFact("", "P17", facts[5], facts[11]),
            RelationFact("", "P17", facts[5], facts[12]),
            RelationFact("", "P17", facts[7], facts[11]),
            RelationFact("", "P17", facts[7], facts[12]),
            RelationFact("", "P159", facts[0], facts[4]),
            RelationFact("", "P159", facts[1], facts[4]),
            RelationFact("", "P159", facts[2], facts[4]),
            RelationFact("", "P127", facts[0], facts[5]),
            RelationFact("", "P127", facts[1], facts[5]),
            RelationFact("", "P127", facts[2], facts[5]),
            RelationFact("", "P159", facts[0], facts[7]),
            RelationFact("", "P159", facts[1], facts[7]),
            RelationFact("", "P159", facts[2], facts[7]),
            RelationFact("", "P17", facts[0], facts[11]),
            RelationFact("", "P17", facts[0], facts[12]),
            RelationFact("", "P17", facts[1], facts[11]),
            RelationFact("", "P17", facts[1], facts[12]),
            RelationFact("", "P17", facts[2], facts[11]),
            RelationFact("", "P17", facts[2], facts[12])
        ])

        self.document = Document(doc_id, text, sentences, tuple(facts))
        self.converter = DocREDConverter()

    def test(self):
        document = list(self.converter.convert(Path("tests/datasets/examples/docred.json")))[0]
        equal_docs(self, self.document, document)
