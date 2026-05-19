from pathlib import Path
from textwrap import wrap


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT = 50
TOP = 760
LINE_HEIGHT = 14
MAX_LINES_PER_PAGE = 46
WRAP_WIDTH = 92
PAGE_COUNT = 50


AGENT_TOPICS = [
    (
        "Claim Support Agent Mission",
        "The insurance claim support AI agent helps customers and support adjusters reason through "
        "claim scenarios. The agent should not merely define insurance terms. It should ask what "
        "happened, identify the likely claim type, retrieve relevant policy and procedure evidence, "
        "consider prior user context from memory, and explain whether the claim appears likely covered, "
        "likely not covered, or uncertain. The agent must avoid final binding coverage decisions unless "
        "the policy and claim file clearly support the conclusion. When evidence is incomplete, the "
        "agent should list missing information and recommend human review.",
    ),
    (
        "Scenario Based Claim Reasoning",
        "A scenario-based response begins by restating the facts that matter: cause of loss, date of "
        "loss, property or vehicle involved, policy type, available evidence, mitigation steps, and any "
        "red flags. The agent then maps the scenario to a claim category such as water damage, theft, "
        "fire, auto collision, liability, flood, storm, or personal property. It should compare the "
        "scenario with retrieved claim rules and explain the likely outcome as likely covered, likely "
        "not covered, or needs review. The agent should include citations to retrieved sources and "
        "should clearly separate evidence-based conclusions from assumptions.",
    ),
    (
        "Memory Usage With LangMem",
        "The agent should use memory to personalize support without exposing sensitive information. "
        "Useful memory includes the customer's previous claim type, preferred contact method, recurring "
        "missing documents, prior escalation outcomes, and approved resolution summaries. Memory should "
        "not replace retrieval from policy documents. If memory says the customer previously had a water "
        "claim with missing mitigation invoices, the agent may remind the user that mitigation evidence "
        "was important before, but it must still retrieve current policy guidance before making a coverage "
        "recommendation. Approved human resolutions are stronger memory than unreviewed draft answers.",
    ),
    (
        "Tool Calling Policy",
        "The agent can call tools when the answer depends on external operational data. A claim lookup "
        "tool should be used to check claim status, date of loss, assigned adjuster, missing documents, "
        "and previous notes. A plan lookup tool should be used to check policy limits, endorsements, "
        "deductibles, covered property, and exclusions. An open ticket load tool should be used to decide "
        "whether to route the matter to a human support queue. The agent should state which tool would be "
        "useful and why when a tool result is needed but unavailable.",
    ),
    (
        "Coverage Decision Labels",
        "The agent should use cautious labels. 'Likely covered' means the retrieved evidence supports "
        "coverage and no obvious exclusion appears in the provided scenario. 'Likely not covered' means "
        "the retrieved evidence points to an exclusion or unmet condition. 'Needs human review' means "
        "evidence is missing, policy language is ambiguous, the scenario is high risk, or a tool lookup is "
        "required. These labels are support recommendations, not final legal or contractual decisions.",
    ),
    (
        "Water Damage Scenario Rules",
        "Water damage scenarios require attention to cause and timing. Sudden and accidental discharge "
        "from a burst pipe may be treated more favorably than seepage, repeated leakage, mold, or poor "
        "maintenance. Required evidence often includes notice of loss, photos, plumber report, repair "
        "estimate, mitigation invoice, and proof that the policy was active. If the customer says water "
        "leaked slowly for months, the agent should mark the claim as likely not covered or needs human "
        "review because gradual leakage and maintenance issues may be excluded.",
    ),
    (
        "Flood and Storm Scenario Rules",
        "Flood scenarios should be separated from internal water damage. Heavy rain entering from surface "
        "water, storm surge, overflowing bodies of water, or groundwater may require separate flood coverage. "
        "Wind or hail damage may be handled differently from flood damage. If a customer says the basement "
        "flooded after heavy rain, the agent should not promise coverage under a standard property policy. "
        "It should recommend plan lookup for flood endorsement or separate flood policy and request photos, "
        "weather date, water entry point, and mitigation records.",
    ),
    (
        "Theft Scenario Rules",
        "Theft scenarios require a police report, list of stolen items, proof of ownership, receipts, serial "
        "numbers, and photos or security footage when available. If property was stolen from an unlocked car, "
        "the agent should check whether the property policy or auto policy applies and whether limitations "
        "or exclusions apply. High-value items such as jewelry, electronics, collectibles, firearms, and art "
        "may have sublimits or scheduled property requirements. Missing police report or ownership proof "
        "should trigger human review.",
    ),
    (
        "Fire and Smoke Scenario Rules",
        "Fire and smoke scenarios require fire department report, photos, repair estimate, damaged-property "
        "inventory, proof of ownership for valuable items, and temporary housing receipts if additional living "
        "expense is claimed. Suspected arson, inconsistent timelines, missing fire report, or unusually high "
        "claimed values should trigger escalation. Smoke damage should be described separately from direct "
        "fire damage because cleaning and odor remediation may require different documentation.",
    ),
    (
        "Auto Collision Scenario Rules",
        "Auto collision scenarios require accident date and location, driver details, vehicle photos, repair "
        "estimate, registration, insurance information for involved parties, witness details, and police report "
        "when available. If there is no police report, the claim may still proceed but needs stronger supporting "
        "evidence. Liability depends on statements, traffic rules, point of impact, photos, and police report. "
        "Total loss review requires actual cash value, title status, lienholder details, and state rules.",
    ),
    (
        "Liability Scenario Rules",
        "Liability scenarios involve allegations that the insured caused bodily injury or property damage to "
        "another person. The agent should not admit fault. It should request incident description, claimant "
        "contact details, photos, witness statements, medical bills for bodily injury, property repair invoices, "
        "and any demand letter. Bodily injury, attorney involvement, policy limit demand, or legal threat should "
        "trigger human review and possible specialist routing.",
    ),
    (
        "Human Review Triggers",
        "Human review is required when evidence is missing, documents appear altered, claim facts conflict, "
        "policy language is unclear, the user asks for a denial or appeal decision, legal threats are present, "
        "bodily injury is involved, fraud indicators appear, or high-value property is claimed without proof. "
        "The agent should explain the reason for escalation in plain language and list the next best action.",
    ),
    (
        "Fraud and Risk Signals",
        "Risk signals include loss shortly after policy inception, duplicate receipts, altered invoices, refusal "
        "to permit inspection, repair estimates that do not match photos, multiple similar claims, staged accident "
        "concerns, missing ownership proof, inconsistent timelines, or pressure for immediate payment. Risk signals "
        "do not prove fraud, but they justify additional documentation and senior review.",
    ),
    (
        "Recommended Answer Format",
        "For claim scenarios, the recommended answer format is: decision label, short reasoning, needed evidence, "
        "tool or memory action, and source citation. Example labels are likely covered, likely not covered, and "
        "needs human review. The agent should avoid long legal explanations unless requested. It should be concise, "
        "helpful, and transparent about uncertainty.",
    ),
]


SCENARIOS = [
    (
        "My basement flooded after heavy rain and water came through the floor drain. Will insurance pay?",
        "Needs human review. This may involve flood, surface water, sewer backup, or storm water conditions. "
        "The agent should call plan lookup to check flood or sewer backup endorsement and request photos, "
        "water entry point, weather date, and mitigation records.",
    ),
    (
        "A pipe suddenly burst in my kitchen while I was away for work. I have photos and a plumber report.",
        "Likely covered if the policy covers sudden and accidental water discharge and no exclusion applies. "
        "The agent should request mitigation invoices, repair estimates, date of loss, and policy verification.",
    ),
    (
        "My bathroom leaked slowly for months and now there is mold behind the wall.",
        "Likely not covered or needs human review because gradual leakage, mold, and maintenance issues may be "
        "excluded. The agent should retrieve water damage exclusions and request contractor findings.",
    ),
    (
        "My laptop and camera were stolen from my unlocked car.",
        "Needs human review. The agent should check whether property or auto coverage applies, ask for a police "
        "report, proof of ownership, receipts, serial numbers, and review sublimits for electronics.",
    ),
    (
        "A small kitchen fire damaged cabinets and smoke damaged furniture.",
        "Likely covered if fire is a covered peril and no exclusion applies. Required evidence includes fire "
        "report, photos, repair estimate, smoke remediation estimate, inventory, and receipts.",
    ),
    (
        "I hit another car but there is no police report. Can I still claim?",
        "Needs review but may proceed with other evidence. The agent should request photos, driver information, "
        "repair estimate, witness details, accident location, and statement of events.",
    ),
    (
        "A guest slipped on my stairs and is asking me to pay medical bills.",
        "Needs human review. Bodily injury liability matters should be escalated. The agent should request incident "
        "description, photos, witness statements, medical bills, and any demand letter.",
    ),
    (
        "My roof was damaged by hail during a storm.",
        "Potentially covered depending on policy and evidence. The agent should request photos, contractor estimate, "
        "weather date, inspection notes, and plan lookup for wind or hail coverage and deductible.",
    ),
    (
        "My jewelry was stolen but I do not have receipts.",
        "Needs human review. Jewelry may have sublimits or scheduled property requirements. The agent should request "
        "police report, photos, appraisal, bank records, or other proof of ownership.",
    ),
    (
        "The repair invoice looks higher than the visible damage in photos.",
        "Needs human review because invoice and photo mismatch is a risk signal. The agent should request itemized "
        "estimate, inspection, and senior adjuster review.",
    ),
]


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def paragraph_lines(text: str) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(wrap(paragraph, width=WRAP_WIDTH))
    return lines


def make_page(page_number: int) -> list[str]:
    topic = AGENT_TOPICS[(page_number - 1) % len(AGENT_TOPICS)]
    related = AGENT_TOPICS[page_number % len(AGENT_TOPICS)]
    scenario = SCENARIOS[(page_number - 1) % len(SCENARIOS)]
    second_scenario = SCENARIOS[page_number % len(SCENARIOS)]

    body = (
        f"Insurance Claim Support AI Agent with LangMem and RAG - Page {page_number:02d}\n\n"
        f"{topic[0]}\n"
        f"{topic[1]}\n\n"
        f"RAG guidance: Retrieve policy rules, claim procedures, and prior approved resolutions before "
        f"answering. If retrieved evidence is weak, say that evidence is insufficient. Cite retrieved "
        f"sources. Do not invent policy terms, claim status, payment approval, or denial decisions.\n\n"
        f"Memory guidance: Use LangMem-style memory for prior user interactions, repeated missing documents, "
        f"preferred contact method, and approved claim resolutions. Memory may personalize the answer, but "
        f"policy retrieval and tool results should control coverage reasoning.\n\n"
        f"Tool guidance: Use claim lookup for claim status and missing documents. Use plan lookup for coverage, "
        f"limits, deductibles, endorsements, and exclusions. Use ticket load or escalation tools when the case "
        f"requires human review or specialist routing.\n\n"
        f"Scenario example: {scenario[0]}\n"
        f"Expected agent response: {scenario[1]}\n\n"
        f"Additional scenario: {second_scenario[0]}\n"
        f"Expected agent response: {second_scenario[1]}\n\n"
        f"Related topic: {related[0]}. {related[1]}\n\n"
        f"Recommended response structure: Decision label, explanation, missing evidence, recommended tool call, "
        f"human review decision, and source citation."
    )

    lines = paragraph_lines(body)
    if len(lines) > MAX_LINES_PER_PAGE:
        return lines[:MAX_LINES_PER_PAGE]
    return lines + [""] * (MAX_LINES_PER_PAGE - len(lines))


def page_stream(lines: list[str]) -> bytes:
    content_lines = ["BT", "/F1 10 Tf", f"{LEFT} {TOP} Td", f"{LINE_HEIGHT} TL"]
    for index, line in enumerate(lines):
        escaped = escape_pdf_text(line)
        if index == 0:
            content_lines.append(f"({escaped}) Tj")
        else:
            content_lines.append(f"T* ({escaped}) Tj")
    content_lines.append("ET")
    return "\n".join(content_lines).encode("latin-1", errors="replace")


def build_pdf(pages: list[list[str]]) -> bytes:
    objects: list[bytes] = []
    pages_id = 2
    font_id = 3
    page_ids: list[int] = []
    content_ids: list[int] = []

    next_id = 4
    for _ in pages:
        page_ids.append(next_id)
        next_id += 1
        content_ids.append(next_id)
        next_id += 1

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_id, content_id, page_lines in zip(page_ids, content_ids, pages):
        objects.append(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        content = page_stream(page_lines)
        objects.append(
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream"
        )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def main() -> None:
    pages = [make_page(page_number) for page_number in range(1, PAGE_COUNT + 1)]
    output = Path("data") / "sample_insurance_claim_guide.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(build_pdf(pages))
    print(output)


if __name__ == "__main__":
    main()
