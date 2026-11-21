//! Unit tests (module `admit_tp_fatal_tests`): source pins for the TP-D rule in `admit_req` — a rank-local
//! staging or device failure during admission must set `fatal` under TP, never skip the lane on one rank only
//! (a one-rank skip splits the lane tables: one rank steps alone, its partner idles). The suite never builds a
//! Scratch (AGENTS 1a-ter), so the rule is pinned on the source text.

    #[test]
    fn every_admit_staging_failure_is_fatal_under_tp() {
        let src = src_text!("exl3_serve");
        let a = &src[src.find("fn admit_req(").expect("admit_req renamed: update this pin")..];
        // sampler, penalties, schema mask, vision engine fault: each arm carries the TP-D fatal
        for anchor in ["sampler staging failed", "penalty staging failed", "schema mask staging failed", "vision setup failed"] {
            let at = a.find(anchor).unwrap_or_else(|| panic!("anchor {anchor:?} renamed: update this pin"));
            let arm = &a[at..];
            let ret = arm.find("return;").expect("the arm returns");
            assert!(arm[..ret].contains("self.fatal = Some("), "{anchor}: the error arm must set fatal under TP before returning");
        }
    }
