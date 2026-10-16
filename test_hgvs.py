from allelion.kb.hgvs import parse_c, parse_p, primary_consequence, refine_consequence


def test_coding_forms():
    c = parse_c("NM_007294.3:c.68_69delAG")
    assert c.transcript == "NM_007294.3" and c.kind == "del" and c.length_change == -2 and c.frameshift
    c = parse_c("c.8754+5G>A")
    assert c.start.offset == 5 and c.splice_distance == 5 and not c.frameshift
    assert parse_c("c.-26G>A").start.region == "utr5" and parse_c("c.-26G>A").start.base == -26
    assert parse_c("c.*105A>G").start.region == "utr3"
    assert parse_c("c.1521_1523delCTT").length_change == -3
    assert parse_c("c.6997_7012del16").length_change == -16
    assert parse_c("c.123_124insA").frameshift
    assert parse_c("not hgvs") is None


def test_protein_forms():
    assert parse_p("p.Phe508del").kind == "inframe_del"
    assert parse_p("p.Arg1443Ter").kind == "nonsense"
    assert parse_p("p.Met1?").kind == "start_loss"
    assert parse_p("p.Ter394GlnextTer?").kind == "stop_loss"
    assert parse_p("p.Glu23ValfsTer17").kind == "frameshift"
    assert parse_p("p.Pro367=").kind == "synonymous"
    assert parse_p("NP_001094.1:p.Ala55Ala=").kind == "synonymous"
    assert parse_p("p.Arg248Gln").short == "R248Q"


def test_consequence_choice_follows_transcript():
    mc = "NM_001170535.2:c.283-3C>T:intron variant;NM_018188.4:c.424C>T:nonsense"
    assert primary_consequence(mc, "NM_018188.4:c.424C>T")[0] == "nonsense"
    assert refine_consequence("intron", parse_c("c.375+1G>A"), None) == "splice_donor"
