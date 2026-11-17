"""Surface forms and question templates.

Why a curated whitelist
-----------------------
FB15k-237 has 237 relations, but ``relation2text.txt`` gives tokenised Freebase
names such as ``"soccer football team current roster. soccer football roster
position position"``. Blindly composing two of those yields questions no human
would ask. We therefore restrict QA generation to relations whose tail is a
concrete, nameable entity, and give each one a hand-written noun phrase. Every
relation outside the whitelist is excluded from generation and reported as such.

The whitelist is a *scope decision*, not a claim that other relations are
invalid: they still exist in the visible graph and remain retrievable.
"""
from __future__ import annotations

# relation -> noun phrase denoting its tail entity, used as "the <phrase> of X"
RELATION_SURFACE: dict[str, str] = {
    # people
    "/people/person/nationality": "country of citizenship",
    "/people/person/place_of_birth": "place of birth",
    "/people/person/profession": "profession",
    "/people/person/gender": "gender",
    "/people/person/religion": "religion",
    # film
    "/film/film/genre": "genre",
    "/film/film/country": "country",
    "/film/film/language": "language",
    "/film/film/produced_by": "producer",
    "/film/film/written_by": "writer",
    "/film/film/music": "composer",
    "/film/film/production_companies": "production company",
    "/film/film/featured_film_locations": "featured location",
    # location
    "/location/country/capital": "capital city",
    "/location/country/form_of_government": "form of government",
    "/location/country/official_language": "official language",
    "/location/location/time_zones": "time zone",
    "/location/location/contains": "contained location",
    # music
    "/music/genre/artists": "artist",
    "/music/artist/origin": "place of origin",
    # tv
    "/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor": "cast member",
    "/tv/tv_program/genre": "genre",
    # organisation / education
    "/education/educational_institution/school_type": "school type",
    "/organization/organization/headquarters./location/mailing_address/citytown":
        "headquarters city",
}

ALLOWED_RELATIONS: frozenset[str] = frozenset(RELATION_SURFACE)

# Ask for the full union across all matching intermediate entities. The same
# templates apply regardless of gold answer count, so wording reveals no
# single-answer/multi-answer label.
TEMPLATES: tuple[str, ...] = (
    "List every {r2} of any {r1} of {head}.",
    "Name every {r2} of any {r1} of {head}.",
    "For {head}, list every {r2} of any {r1}.",
    "Return every {r2} of any {r1} of {head}.",
)

# Aliases retain compatibility with the builder's existing imports.
TEMPLATES_SINGULAR = TEMPLATES
TEMPLATES_PLURAL = TEMPLATES

TEMPLATE_VERSION = "v4"


def relation_surface(relation: str) -> str:
    """Human noun phrase for a relation; falls back to its last path segment."""
    phrase = RELATION_SURFACE.get(relation)
    if phrase:
        return phrase
    tail = relation.rsplit("/", 1)[-1].replace("_", " ").strip()
    return tail or relation


def entity_surface(entity_id: str, entity_text: dict[str, str]) -> str:
    """Readable name for an entity, falling back to the raw Freebase id."""
    name = entity_text.get(entity_id, "").strip()
    return name or entity_id


def render_question(
    head: str,
    relation_path: tuple[str, ...],
    template: str,
    entity_text: dict[str, str],
) -> str:
    """Fill one template for a two-hop path."""
    first, second = relation_path
    return template.format(
        head=entity_surface(head, entity_text),
        r1=relation_surface(first),
        r2=relation_surface(second),
    )
