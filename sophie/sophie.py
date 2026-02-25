# SOPHIE Voice Assistant v3.6
# NLP-Upgrade: Sentence-Transformer Intent-Engine (paraphrase-multilingual-MiniLM-L12-v2)
# CUDA → CPU Fallback, Vierschicht-Architektur:
#   1a. ASR-Fuzzy-Korrektur (Levenshtein + Whisper-Fehlerwörterbuch)
#   1b. Regex-Vorfilter (Mikrosekunden)
#   2.  Sentence-Transformer Embedding-Matching (threshold 0.50)
#   3.  Confidence-Fallback → LLM
# Whisper medium für bessere Transkription
# api_fallback als expliziter Intent mit Smalltalk-Beispielen
# Verbesserte Notiz-Inhaltsextraktion (nämlich, neue X nämlich dass...)
# Mehr Beispielsätze für robusteres ST-Matching
# v3.4: Smalltalk-Intents (lokal ohne LLM), Timer-Alarm an App-Clients,
#        Multi-Client per-IP Tracking, mehr Intent-Beispiele
# v3.5: Device Groups – mehrere Geräte agieren als eines
#        (gemeinsames Wakeword-Listening, Peer-Muting, Sync-TTS)
# v3.6: Persistente Timer/Wecker – absolute Zielzeit in JSON-Datei (data/timers.json)
#        statt asyncio.sleep-Countdown. Übersteht Reconnects (3h-Zyklus).
#        Per-Owner Isolation: nur der Client/die Gruppe die den Timer gestellt hat
#        hört den Alarm. Andere Clients/Gruppen sind nicht betroffen.

CONFIG = {
    "wake_model_path":   "model/sofie_20260213_173731.onnx",
    "wake_threshold":    0.2,
    "wake_chunk_size":  1280,
    "cmd_model":        "small",
    "cmd_language":     "de",
    "cmd_silence_secs":  1.2,
    "cmd_max_secs":      14,
    "cmd_min_secs":       0.5,
    "speech_rms_threshold": 0.010,
    "tts_api_url":      "http://localhost:8020/tts_to_audio/",
    "tts_speaker":      "1",
    "tts_language":     "de",
    "tts_speed":         1.0,
    "api_url":          "http://localhost:8002/api/v1/generate",
    "api_timeout":       90,
    "api_max_length":    50,
    "api_history_turns": 2,
    "api_system_prompt": (
        "Du bist Sophie, ein KI Assistent. "
        "Antworte immer auf Deutsch, kurz und direkt, maximal 1 Satz. "
        "Kein Markdown, keine Listen, nur natürliche gesprochene Sprache. Du bist frech auch sarkastisch, aber nicht unhöflich."
        "Beginne die Antwort nie mit Sophie: oder ähnlichem."
    ),
    "alarm_sound":       "wecker.wav",
    "alarm_loop_secs":   60,
    "timer_reminder_interval": 600,
    "weather_lat":       50.110924,
    "weather_lon":        8.682127,
    "weather_city":      "Frankfurt",
    "sample_rate":       16000,
    "intent_threshold":  0.25,
    "host":     "0.0.0.0",
    "port":      8765,
    "app_http_port":  8775,
    "app_audio_in_port":  8776,
    "app_audio_out_port": 8777,
    "static_dir": "static",
    "log_level":  "INFO",
    "db_path":    "data/kalender.json",
    "notes_path": "data/notizen.json",
    "todo_path":  "data/todo.json",
    "reminders_path": "data/erinnerungen.json",
    "timers_path":    "data/timers.json",
    "witze_path":     "data/witze.txt",
    "audio_device_id": 6,
    # ── Device Groups: Geräte die als ein Gerät agieren ──────────────
    # Alle IPs in einer Gruppe hören gleichzeitig auf das Wakeword.
    # Erkennt eins das Keyword, hört nur dieses zu (die anderen pausieren).
    # Thinking-Beeps und TTS gehen an ALLE Geräte der Gruppe (sync).
    "device_groups": [
        {"name": "Wohnzimmer", "ips": ["192.168.188.150", "192.168.188.151"]},
    ],
}

INTENTS = [

    {"name": "kalender_abfrage", "beispiele": [
        "was habe ich am montag", "was steht am dienstag an", "zeig mir meine termine",
        "was sind meine termine am zwölften", "welche termine habe ich am freitag",
        "habe ich am donnerstag etwas", "was ist für den fünfzehnten geplant",
        "zeige termine für den dritten februar", "was passiert am dritten",
        "was habe ich am nächsten montag", "termine am wochenende",
        "was steht diese woche an", "gibt es termine am ersten",
        "kalender für morgen zeigen", "terminübersicht für nächste woche",
        "wann ist mein nächster termin", "was habe ich diese woche",
        "habe ich was am samstag", "was steht am zwanzigsten an",
        "was steht in meinem kalender",
        "was steht heute im kalender",
        "was steht in meinem kalender drin",
        "sag mir was im kalender steht",
        "was habe ich am mittwoch vor", "was liegt am sonntag an",
        "habe ich am freitag abend was", "welche termine stehen an",
        "was gibt es am dienstag", "ist am samstag was geplant",
        "was habe ich am wochenende vor", "kalender abfragen",
        "schau in meinen kalender", "was ist nächste woche los",
        "habe ich übermorgen einen termin",
    ]},
    {"name": "kalender_eintragen", "beispiele": [
        "trag einen termin ein", "fuege einen termin hinzu", "neuer termin am montag",
        "termin erstellen für dienstag", "merke dir arzttermin am freitag",
        "schreib auf dass ich am montag zahnarzt habe", "speichere termin am dritten",
        "termin hinzufügen am zwölften", "notiere meeting am donnerstag",
        "trag für morgen einkaufen ein", "erstelle einen kalendereintrag",
        "ich habe am samstag hochzeit", "trage arzttermin für freitag ein",
        "mache einen eintrag für den zweiten mai", "neuer kalendereintrag",
        "termin am ersten märz eintragen", "kalender eintrag erstellen",
        "trag ein dass ich am dreizehnten februar zähne putzen muss",
        "trage in den kalender ein dass", "neuen termin anlegen",
        "schreib in den kalender dass ich am montag meeting habe",
        "füge zum kalender hinzu", "setze einen termin für morgen",
        "kalender eintrag für nächsten mittwoch", "termin am samstag notieren",
        "trag bitte einen termin ein", "mach einen termin am freitag",
        "speichere im kalender", "neuer eintrag am donnerstag",
        "ich muss am dienstag zum arzt eintragen",
        "pack in den kalender dass ich am mittwoch frei habe",
        "eintrag erstellen für den zwanzigsten",
    ]},
    {"name": "kalender_löschen", "beispiele": [
        "lösche den termin am montag", "entferne eintrag vom dienstag",
        "streiche den zahnarzt am freitag", "termin absagen am dritten",
        "lösch den eintrag", "entferne meinen termin am samstag",
        "kalender eintrag löschen", "meeting am donnerstag streichen",
        "termin vom dreizehnten entfernen", "cancel termin am zehnten",
        "lösche alle einträge am dienstag", "lösche meinen kalender",
        "kalendereinträge löschen", "lösche die einträge",
        "entferne alle termine am montag", "lösche alles am freitag",
        "kalender für heute löschen", "lösche alle kalendereinträge",
        "lösche alle einträge für heute", "entferne den eintrag am dienstag",
        "streiche alle termine am donnerstag", "lösche alle termine",
        "lösche den kalender am montag", "meinen kalender löschen",
    ]},
    {"name": "heute_termine", "beispiele": [
        "was habe ich heute", "meine heutigen termine", "was steht heute an",
        "gibt es heute etwas", "was ist heute geplant", "habe ich heute einen termin",
        "was passiert heute", "zeig mir was heute ansteht", "termine für heute",
        "was muss ich heute noch machen", "habe ich heute was",
        "was habe ich heute auf dem plan", "was steht heute im kalender",
        "stehen heute termine an", "ist heute was los",
        "sag mir meine termine für heute", "heutige termine bitte",
        "was liegt heute an", "was geht heute", "kalender heute",
    ]},
    {"name": "nächste_termine", "beispiele": [
        "was kommt als nächstes", "meine nächsten termine", "zeig alle termine",
        "kommende termine", "was steht bald an", "was habe ich in den nächsten tagen",
        "terminübersicht", "alle termine zeigen", "was habe ich diese woche noch",
        "bevorstehende termine", "kalenderübersicht", "alle meine einträge",
        "zeig mir alle termine", "nächste termine anzeigen",
    ]},
    {"name": "notiz_hinzufuegen", "beispiele": [
        "mach eine notiz", "notiere das", "schreib das auf", "merke dir das",
        "füge notiz hinzu", "neue notiz", "notiz erstellen",
        "schreib auf dass ich morgen essen kaufen muss",
        "mache eine notiz dass ich morgen einkaufen muss",
        "notiere dass ich morgen zum arzt muss",
        "schreib auf ich muss morgen anrufen",
        "mach dir eine notiz", "kannst du dir merken",
        "vergiss nicht dass ich morgen", "merke dir bitte",
        "ich möchte eine notiz erstellen", "notiz anlegen",
        "mache eine notiz dass ich mir die hände waschen muss",
        "notiere dass ich essen kaufen muss",
        "merk dir dass ich morgen früh aufstehen muss",
        "ich will eine notiz machen", "kannst du das notieren",
        # neue erweiterte Beispiele
        "füge eine neue notiz hinzu nämlich dass flo ein arschloch ist",
        "mache eine neue notiz nämlich dass ich morgen arbeiten muss",
        "notiz hinzufügen dass ich zahnarzt habe",
        "schreib dir auf dass ich peter anrufen muss",
        "bitte notiere dass ich das formular ausfüllen muss",
        "merke dir bitte dass ich heute noch einkaufen muss",
        "zu meinen notizen hinzufügen dass ich flo mag",
        "erstelle eine notiz mit dem inhalt ich muss sport machen",
        "speicher dass ich morgen früh aufstehen muss",
        "trag als notiz ein dass ich die miete bezahlen muss",
    ]},
    {"name": "notizen_lesen", "beispiele": [
        "was sind meine notizen", "zeig meine notizen", "lies notizen vor",
        "welche notizen habe ich", "meine notizen", "alle notizen anzeigen",
        "was habe ich notiert", "notizen vorlesen", "zeig mir meine notizen",
        "was habe ich mir gemerkt", "zeig notizen", "notizen abfragen",
        # neue Beispiele
        "lies mir alle notizen vor", "les mir die notizen vor",
        "zeig mir alle meine notizen", "was steht in meinen notizen",
        "welche notizen habe ich gemacht", "vorlesen der notizen",
        "kannst du mir meine notizen vorlesen", "meine notizen bitte",
        "was habe ich alles notiert", "alle meine notizen anzeigen",
        "habe ich notizen", "gibt es notizen",
        "was hab ich mir aufgeschrieben", "was hast du dir gemerkt",
        "notizen anzeigen bitte", "sag mir meine notizen",
        "welche notizen gibt es", "zeig was ich notiert habe",
    ]},
    {"name": "notiz_löschen", "beispiele": [
        "lösche alle notizen", "notizen löschen", "alle notizen entfernen",
        "notiz löschen", "notizen leeren", "lösch meine notizen",
        "alle notizen löschen", "lösche die notiz über",
        "entferne die notiz", "lösch notiz",
        # neue Beispiele
        "lösche meine notizen", "entferne alle notizen",
        "notizen komplett löschen", "leere alle notizen",
        "lösch alle meine notizen", "notizen entfernen",
        "alle notizen raus", "notizliste leeren",
        "entferne notiz über einkaufen", "lösche die notiz mit sport",
    ]},
    {"name": "timer_stellen", "beispiele": [
        "timer stellen", "setze einen timer", "starte einen timer",
        "timer auf fünf minuten", "timer für zwei minuten",
        "stelle einen timer für zehn minuten", "timer zwei minuten",
        "wecker auf dreißig sekunden", "erinnere mich in fünf minuten",
        "wecke mich in zehn minuten", "alarm in einer stunde",
        "stell einen timer auf eine minute", "setze timer für drei minuten",
        "timer dreißig minuten bitte", "alarm auf fünf minuten setzen",
        "timer für neunzig sekunden", "timer fünfzehn minuten",
        "stell mir einen timer", "einen timer auf zwei minuten",
        "timer auf eine stunde", "erinnere mich in zwei stunden",
        "starte countdown zwei minuten", "drei minuten timer",
        "kannst du einen timer für zwei minuten stellen",
        "timer auf sieben minuten", "zwanzig minuten timer bitte",
        "countdown auf drei minuten starten", "weck mich in einer stunde",
    ]},
    {"name": "timer_stoppen", "beispiele": [
        "timer stoppen", "alarm aus", "wecker aus", "stopp den timer",
        "timer abbrechen", "alarm stoppen", "timer beenden",
        "mach den alarm aus", "timer löschen", "countdown stoppen",
        "wecker abschalten", "abbrechen", "alarm abbrechen",
        "mach den wecker aus", "stoppe alarm", "timer off",
        "sei still", "hör auf", "stopp", "ruhig bitte",
        "stopp den alarm", "mach das aus", "timer stopp",
        "alarm abschalten", "piepen aufhören", "aufhören",
        "bitte sei still", "hör auf zu piepen", "mach den piepser aus",
        "mach den ton aus", "alarm deaktivieren", "wecker stoppen",
        "halt die Klappe", "halts maul", "Klappe", "ruhe",
        # neue Beispiele - damit "lösche meine X" nicht hier landet
        "stopp bitte", "alarm bitte aus", "das piepen stoppen",
        "ich will ruhe", "mach den krach aus",
    ]},
    {"name": "timer_abfragen", "beispiele": [
        "wie lange läuft der timer noch", "wie viel zeit ist noch auf dem timer",
        "wie lange noch", "timer restzeit", "wie lange hat der timer noch",
        "wann klingelt der timer", "noch wie lange", "timer status",
        "wie lange läuft der noch", "wieviel zeit noch", "timer check",
        "wie lange noch beim timer", "wie viel minuten noch",
    ]},
    {"name": "wecker_stellen", "beispiele": [
        "stelle einen wecker um zwölf uhr fünfzehn",
        "wecker um sieben uhr", "stell einen wecker auf acht uhr",
        "wecker für morgen um sechs", "weck mich um sieben",
        "wecker auf halb acht", "stell mir einen wecker um neun",
        "wecker um viertel nach sechs", "wecke mich um fünf uhr dreißig",
        "wecker um achtzehn uhr", "stell wecker auf zwanzig uhr",
        "wecker stellen um sieben uhr dreißig",
        "stelle einen wecker um sechs uhr morgens",
        "wecker für halb sieben", "wecker auf viertel vor acht",
        "weck mich morgen um acht", "wecker um neun uhr fünfundvierzig",
        "stell einen wecker um dreizehn uhr", "wecker auf zwölf",
        "wecker um fünf uhr achtzehn", "stell wecker um elf",
        "ich brauche einen wecker um sieben", "mach einen wecker um sechs",
        "wecker um halb neun stellen", "setze einen wecker auf zehn uhr",
        "wecker morgen früh um sechs", "wecker für heute abend um acht",
    ]},
    {"name": "uhrzeit", "beispiele": [
        "wie spät ist es", "wieviel uhr ist es", "sag mir die uhrzeit",
        "aktuelle uhrzeit", "wie viel uhr haben wir", "uhrzeit bitte",
        "wie viel uhr ist es gerade", "was ist die uhrzeit",
        "sag mir wie spät es ist", "wie spät haben wir es",
        "wie viel uhr haben wir gerade", "wie spät ist es gerade",
        "sag die uhrzeit", "uhrzeit ansagen",
    ]},
    {"name": "datum_heute", "beispiele": [
        "welches datum ist heute", "was ist heute für ein datum",
        "welcher tag ist heute", "was für ein tag ist heute",
        "welcher wochentag ist heute", "datum heute",
        "was ist heute für ein tag", "sag mir das datum",
        "welches datum haben wir heute", "der wievielte ist heute",
        "was haben wir für ein datum", "welches datum schreiben wir",
        "welchen tag haben wir gerade", "welchen wochentag haben wir",
        "was für ein datum ist übermorgen", "datum übermorgen",
        "was ist in drei tagen für ein datum", "datum in fünf tagen",
        "welcher monat ist gerade", "welches jahr haben wir", "datum bitte",
        "welches datum ist morgen",
    ]},
    {"name": "wetter", "beispiele": [
        "wie ist das wetter", "wie wird das wetter", "ist es draussen kalt",
        "was für wetter haben wir", "wetter heute", "wetter morgen",
        "wie kalt ist es draussen", "wie warm ist es", "brauche ich eine jacke",
        "regnet es heute", "wie ist die temperatur", "temperatur draussen",
        "wie warm wird es heute", "wetter vorhersage", "wird es regnen",
        "wetter für heute", "wetter in weilburg", "wetter aktuell",
        "wie viel grad hat es", "wetter bericht", "wie ist das wetter draussen",
        "brauche ich einen regenschirm", "scheint die sonne",
    ]},
    {"name": "erinnerung_setzen", "beispiele": [
        "erinnere mich heute um achtzehn uhr dass ich müll rausbringen muss",
        "erinnere mich morgen um neun uhr an den arzt",
        "setze eine erinnerung für morgen früh",
        "erinnerung erstellen für heute abend",
        "erinnere mich um fünfzehn uhr",
        "stell eine erinnerung für morgen um acht",
        "erinnere mich in einer stunde",
        "mach eine erinnerung für heute um zwanzig uhr",
        "erinnerung setzen für heute um siebzehn uhr dreißig",
        "ich brauche eine erinnerung für morgen",
        "erinnerung für heute abend erstellen",
        "erinnere mich um halb sieben dass ich sport machen soll",
        "setze erinnerung auf morgen acht uhr",
        "erinnere mich bitte um drei uhr", "erinnere mich in dreißig minuten",
        "erinnere mich um fünf uhr achtzehn",
        "mache eine erinnerung für heute um fünf",
        "neue erinnerung für morgen um zehn",
        "kannst du mich um sieben erinnern",
        "sag mir bescheid um acht uhr",
    ]},
    {"name": "erinnerungen_lesen", "beispiele": [
        "welche erinnerungen habe ich", "zeig meine erinnerungen",
        "was sind meine erinnerungen", "alle erinnerungen anzeigen",
        "meine erinnerungen", "gibt es erinnerungen",
        "erinnerungen anzeigen", "welche erinnerungen sind gesetzt",
        "zeig alle erinnerungen",
    ]},
    {"name": "erinnerung_löschen", "beispiele": [
        "lösche alle erinnerungen", "erinnerungen löschen",
        "entferne alle erinnerungen", "erinnerung löschen",
        "alle erinnerungen entfernen", "lösch meine erinnerungen",
    ]},
    {"name": "todo_hinzufuegen", "beispiele": [
        "setze auf meine To-do  liste, dass ich kacken gehen muss",
        "füge zur To-do liste hinzu", "To-do  hinzufügen",
        "setze auf die To-do  liste dass ich einkaufen muss",
        "auf die To-do  liste schreiben", "To-do  erstellen",
        "füge aufgabe hinzu", "neue aufgabe", "aufgabe hinzufügen",
        "trag auf die To-do  liste", "schreib auf die To-do  liste",
        "To-do  liste ergänzen", "aufgabe zur liste hinzufügen",
        "neue To-do  aufgabe", "zur aufgabenliste hinzufügen",
        "schreibe auf meine todo liste dass",
        "schreib auf die todo liste dass ich einkaufen muss",
        "packe auf die todo liste", "auf meine todo liste schreiben",
    ]},
    {"name": "todo_lesen", "beispiele": [
        "was steht auf meiner To-do  liste", "zeig meine To-do  liste",
        "meine aufgaben", "To-do  liste anzeigen", "was muss ich noch machen",
        "welche aufgaben habe ich", "To-do  liste vorlesen",
        "zeig mir meine To-do s", "aufgabenliste anzeigen",
        "was sind meine To-do s", "alle aufgaben zeigen",
        "lies meine To-do  liste vor",
    ]},
    {"name": "todo_löschen", "beispiele": [
        "entferne kacken von der To-do  liste", "lösche aufgabe von der liste",
        "To-do  entfernen", "aufgabe abgehakt", "aufgabe erledigt",
        "entferne von der To-do  liste", "lösch das von der To-do  liste",
        "aufgabe von der liste streichen", "erledigt", "hak ab",
        "entferne einkaufen von der To-do  liste",
        "lösche die aufgabe", "aufgabe löschen",
        "von der liste nehmen",
    ]},
    {"name": "witz", "beispiele": [
        "erzähl mir einen witz", "mach einen witz", "sag einen witz",
        "hast du einen witz", "witz bitte", "ich will einen witz hören",
        "erzähl was lustiges", "sag was lustiges", "mach mal einen witz",
        "erzähl einen witz", "kannst du einen witz erzählen",
        "einen witz bitte", "ich brauche einen witz",
        "witz", "witze", "hast du witze", "erzähle einen witz",
        "mach mich zum lachen", "bring mich zum lachen",
        "sag mir einen witz", "weißt du einen witz",
        "kennst du einen witz", "kennst du witze",
        "hast du was lustiges", "erzähl mir was lustiges",
        "ich will lachen", "etwas lustiges bitte",
        "sag mal was lustiges", "hast du einen guten witz",
    ]},
    {"name": "hilfe", "beispiele": [
        "was kannst du", "hilfe", "was sind deine funktionen",
        "womit kannst du mir helfen", "zeige was du kannst",
        "was kann sophie", "wie funktionierst du",
        "erkläre deine funktionen", "was für befehle gibt es",
        "befehlsübersicht", "ich brauche hilfe", "wie nutze ich dich",
        "welche commands gibt es", "zeig mir deine fähigkeiten",
    ]},
    {"name": "mathe", "beispiele": [
        "was ist zwei plus zwei", "rechne zehn mal fünf", "was ist drei mal sieben",
        "was ergibt fünf plus acht", "berechne zwanzig geteilt durch vier",
        "was ist hundert minus dreißig", "rechne mal aus was 123 plus 32 mal 3 ist",
        "was ergibt 5 hoch 2", "berechne die wurzel von 144", "was ist 17 mal 23",
        "wie viel ist 1000 geteilt durch 8", "rechne 50 plus 25 minus 10",
        "was ist 99 mal 99", "berechne 256 geteilt durch 16",
        "wie viel ergibt 42 plus 58", "was ist 15 prozent von 200",
        "rechne aus 7 mal 8 plus 3", "was ergibt 500 minus 123",
        "berechne 12 hoch 3", "was ist die quadratwurzel von 225",
        "was ist eins plus eins", "was ist drei plus vier",
        "was ist 3.14 mal 2", "rechne 1024 durch 32",
        "was ergibt 7 fakultät", "berechne sinus von 90",
        "was ist ein drittel von 99", "wie viel ist 33 prozent von 600",
        "rechne 2 hoch 10", "was ist 45 plus 67 mal 2",
    ]},
    # ── Smalltalk-Intents (lokal, NICHT via LLM) ─────────────────────────
    {"name": "smalltalk_begruessung", "beispiele": [
        "hallo", "hallo sophie", "hey", "hey sophie", "hi", "hi sophie",
        "guten morgen", "guten tag", "guten abend", "moin", "moin moin",
        "na", "na du", "servus", "grüß dich", "grüß gott",
        "huhu", "hallöchen", "halli hallo", "was geht",
        "guten morgen sophie", "morgen sophie", "abend sophie",
        "hallo wie gehts", "hey was geht",
    ]},
    {"name": "smalltalk_verabschiedung", "beispiele": [
        "tschüss", "tschüss sophie", "auf wiedersehen", "bis später",
        "bis dann", "ciao", "bye", "bye bye", "gute nacht",
        "schlaf gut", "bis morgen", "man sieht sich", "mach's gut",
        "ich gehe jetzt", "ich bin dann weg", "bis zum nächsten mal",
        "adieu", "auf wiederhören", "schönen abend noch",
        "gute nacht sophie", "tschau", "tschö",
    ]},
    {"name": "smalltalk_befinden", "beispiele": [
        "wie geht es dir", "wie geht es dir heute", "wie geht's",
        "wie gehts dir", "wie geht's dir so", "alles klar bei dir",
        "wie fühlst du dich", "geht es dir gut", "und dir",
        "wie läufts", "was machst du gerade", "was machst du so",
        "wie ist deine laune", "bist du gut drauf",
        "wie gehts dir heute", "alles fit bei dir",
        "was geht bei dir", "und wie geht es dir",
        "bist du müde", "langweilst du dich",
    ]},
    {"name": "smalltalk_identitaet", "beispiele": [
        "wer bist du", "wie heißt du", "was bist du",
        "wie ist dein name", "hast du einen namen",
        "bist du eine ki", "bist du ein roboter", "bist du echt",
        "bist du eine künstliche intelligenz", "bist du ein mensch",
        "bist du ein bot", "bist du ein computer",
        "was für eine ki bist du", "bist du real",
        "sag mir deinen namen", "wie darf ich dich nennen",
        "hast du einen nachnamen", "wie heißt du mit nachnamen",
        "hast du einen spitznamen", "wer oder was bist du",
        "beschreib dich mal", "stell dich vor",
    ]},
    {"name": "smalltalk_ersteller", "beispiele": [
        "wer hat dich erstellt", "wer hat dich gemacht",
        "wer hat dich programmiert", "wer hat dich gebaut",
        "wer hat dich entwickelt", "wer ist dein ersteller",
        "wer ist dein schöpfer", "wer ist dein entwickler",
        "wer hat dich erschaffen", "wer hat dich geschrieben",
        "von wem wurdest du erstellt", "von wem wurdest du programmiert",
        "wer ist dein vater", "wer ist dein papa",
        "wer ist für dich verantwortlich", "wer steckt hinter dir",
        "wer hat dich ins leben gerufen", "wer ist dein creator",
        "wer hat dich designed", "wer ist dein erfinder",
        "woher kommst du", "wo wurdest du entwickelt",
    ]},
    {"name": "smalltalk_alter", "beispiele": [
        "wie alt bist du", "wann wurdest du geboren",
        "wann wurdest du erstellt", "wann wurdest du gemacht",
        "wie lange gibt es dich", "seit wann gibt es dich",
        "wann bist du entstanden", "wie lange existierst du",
        "hast du ein alter", "hast du einen geburtstag",
        "wann hast du geburtstag", "in welchem jahr wurdest du erstellt",
        "wie alt bist du eigentlich", "bist du neu",
        "seit wann bist du online",
    ]},
    {"name": "smalltalk_danke", "beispiele": [
        "danke", "dankeschön", "danke schön", "danke dir",
        "danke sophie", "vielen dank", "besten dank",
        "ich danke dir", "danke sehr", "merci", "thanks",
        "das war hilfreich", "super danke", "danke dafür",
        "toll danke", "perfekt danke", "cool danke",
    ]},
    {"name": "smalltalk_kompliment", "beispiele": [
        "du bist toll", "du bist super", "du bist die beste",
        "ich mag dich", "du bist cool", "du bist lustig",
        "du bist schlau", "du bist nett", "gut gemacht",
        "das hast du gut gemacht", "nice", "geil", "läuft bei dir",
        "du bist witzig", "nicht schlecht", "beeindruckend",
        "du bist echt gut", "respekt", "stark",
    ]},
    {"name": "smalltalk_beleidigung", "beispiele": [
        "du bist doof", "du bist blöd", "du nervst",
        "du bist scheiße", "du bist nutzlos", "du kannst nichts",
        "du bist dumm", "idiot", "du taugst nichts",
        "du bist hässlich", "du bist peinlich", "du stinkst",
        "du bist der letzte dreck", "du kannst gar nichts",
        "du bist unnütz", "versager", "du bist beschissen",
        "fick dich", "verpiss dich", "leck mich",
    ]},
    {"name": "smalltalk_gefuehle", "beispiele": [
        "ich bin traurig", "ich bin müde", "ich bin gelangweilt",
        "ich langweile mich", "mir ist langweilig",
        "ich bin sauer", "ich bin wütend", "ich bin genervt",
        "ich bin glücklich", "ich bin fröhlich",
        "ich bin einsam", "ich fühle mich allein",
        "ich habe angst", "ich bin gestresst",
        "ich bin happy", "mir geht es schlecht",
        "mir geht es gut", "ich bin gut drauf",
        "ich bin schlecht drauf", "ich habe schlechte laune",
    ]},
    # api_fallback: Wissensfragen, komplexe Fragen → an LLM
    {"name": "api_fallback", "beispiele": [
        "wie backe ich kuchen", "erkläre mir quantenphysik",
        "was ist die hauptstadt von frankreich", "wer hat deutschland gegründet",
        "übersetze hallo auf englisch", "wie sagt man danke auf spanisch",
        "schreib mir ein gedicht", "erzähl mir eine geschichte",
        "was denkst du darüber", "was ist deine meinung",
        "wie wird das wochenende werden", "was empfiehlst du mir",
        "kannst du mir helfen",
        "ich habe eine frage", "kannst du mir erklären",
        "was passiert wenn", "warum ist der himmel blau",
        "was ist künstliche intelligenz", "erkläre machine learning",
        "was ist der unterschied zwischen", "vergleiche",
        "ich brauche einen rat", "was soll ich machen",
        "erzähl mir etwas über", "was weißt du über",
        "unterhalte mich", "sag mir was interessantes",
        "erzähl mir was", "was weißt du", "hast du eine meinung",
    ]},
]

import asyncio
import base64
import json
import logging
import os
import queue
import random
import re
import sys
import tempfile
import threading
import time
import sounddevice as sd
import soundfile as sf
import io  # Wird für Byte-Streams benötigt
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Set

import numpy as np

logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"]),
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sophie")

_cmd_model = None
_cmd_model_lock = threading.Lock()

# ── Whisper-Queue: ein Modell, serialisierte Transkriptions-Jobs ──────────────
import queue as _queue

_whisper_queue: _queue.Queue = _queue.Queue()
_whisper_worker_started = False
_whisper_worker_lock = threading.Lock()

def _whisper_worker():
    """Einzelner Worker-Thread: ladet Modell einmal, verarbeitet Jobs sequenziell."""
    model = get_cmd_model()
    log.info("Whisper-Queue-Worker bereit.")
    while True:
        samples, event, result_box = _whisper_queue.get()
        try:
            result_box["result"] = _do_transcribe(model, samples)
        except Exception as e:
            log.error(f"Whisper-Worker Fehler: {e}")
            result_box["result"] = ""
        finally:
            event.set()

def _ensure_whisper_worker():
    global _whisper_worker_started
    with _whisper_worker_lock:
        if not _whisper_worker_started:
            _whisper_worker_started = True
            t = threading.Thread(target=_whisper_worker, daemon=True, name="whisper-worker")
            t.start()
_tts = None
_chat_history: list = []
_chat_history_lock = threading.Lock()

MONATE = {
    "januar": 1, "jänner": 1, "februar": 2, "märz": 3, "april": 4,
    "mai": 5, "juni": 6, "juli": 7, "august": 8, "september": 9,
    "oktober": 10, "november": 11, "dezember": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "dez": 12,
    # umlaute
    "m\u00e4rz": 3,
}
MONATE_DE = {1:"Januar",2:"Februar",3:"M\u00e4rz",4:"April",5:"Mai",6:"Juni",
             7:"Juli",8:"August",9:"September",10:"Oktober",11:"November",12:"Dezember"}
WOCHENTAGE_DE = ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag","Samstag","Sonntag"]

_WORT_ZU_ZAHL = {
    "null": 0, "eins": 1, "eine": 1, "einem": 1, "einen": 1, "ein": 1, "halb": 0.5,
    "anderthalb": 1.5, "eineinhalb": 1.5, "anderthalbe": 1.5,
    "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6,
    "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
    "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14,
    "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17, "achtzehn": 18,
    "neunzehn": 19, "zwanzig": 20, "einundzwanzig": 21, "zweiundzwanzig": 22,
    "dreiundzwanzig": 23, "vierundzwanzig": 24, "fünfundzwanzig": 25,
    "sechsundzwanzig": 26, "siebenundzwanzig": 27, "achtundzwanzig": 28,
    "neunundzwanzig": 29,
    "dreißig": 30, "einunddreißig": 31, "vierzig": 40, "fünfzig": 50,
    "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90,
    "hundert": 100, "tausend": 1000, "million": 1000000,
    # mit Umlauten
    "f\u00fcnf": 5, "zw\u00f6lf": 12, "f\u00fcnfzehn": 15,
    "drei\u00dfig": 30, "f\u00fcnfzig": 50, "f\u00fcnfundzwanzig": 25,
}

_STUNDEN_WORT = {
    "null": 0, "ein": 1, "eins": 1, "eine": 1, "zwei": 2, "drei": 3, "vier": 4,
    "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
    "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15,
    "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
    "zwanzig": 20, "einundzwanzig": 21, "zweiundzwanzig": 22, "dreiundzwanzig": 23,
    "f\u00fcnf": 5, "zw\u00f6lf": 12, "f\u00fcnfzehn": 15,
    "achtzehn": 18,
}

_MINUTEN_WORT = {
    "null": 0, "fünf": 5, "zehn": 10, "fünfzehn": 15, "zwanzig": 20,
    "fünfundzwanzig": 25, "dreißig": 30, "fünfundreißig": 35,
    "vierzig": 40, "fünfundvierzig": 45, "fünfzig": 50, "fünfundfünfzig": 55,
    "f\u00fcnf": 5, "f\u00fcnfzehn": 15, "drei\u00dfig": 30,
}


def _zahlen_normalisieren(text: str) -> str:
    t = text
    for wort, zahl in sorted(_WORT_ZU_ZAHL.items(), key=lambda x: -len(x[0])):
        t = re.sub(r'\b' + re.escape(wort) + r'\b', str(zahl), t)
    return t


def parse_uhrzeit(text: str) -> Optional[tuple]:
    t = text.lower().strip()
    # halb X
    m = re.search(r'\bhalb\s+(\w+)', t)
    if m:
        sw = m.group(1)
        st = _STUNDEN_WORT.get(sw) if not sw.isdigit() else int(sw)
        if st is not None and 1 <= st <= 24:
            return (st - 1) % 24, 30
    # viertel nach X
    m = re.search(r'\bviertel\s+nach\s+(\w+)', t)
    if m:
        sw = m.group(1)
        st = _STUNDEN_WORT.get(sw) if not sw.isdigit() else int(sw)
        if st is not None:
            return st % 24, 15
    # viertel vor X
    m = re.search(r'\bviertel\s+vor\s+(\w+)', t)
    if m:
        sw = m.group(1)
        st = _STUNDEN_WORT.get(sw) if not sw.isdigit() else int(sw)
        if st is not None:
            return (st - 1) % 24, 45
    # X Uhr Y (stunde + minuten)
    m = re.search(r'\b(\w+)\s+uhr\s+(\w+)', t)
    if m:
        h_raw, min_raw = m.group(1), m.group(2)
        st = _STUNDEN_WORT.get(h_raw) if not h_raw.isdigit() else int(h_raw)
        mi = _MINUTEN_WORT.get(min_raw) if not min_raw.isdigit() else int(min_raw)
        if st is not None and mi is not None and 0 <= st <= 23 and 0 <= mi <= 59:
            return st, mi
    # X Uhr (nur stunde)
    m = re.search(r'\b(\w+)\s+uhr\b', t)
    if m:
        h_raw = m.group(1)
        st = _STUNDEN_WORT.get(h_raw) if not h_raw.isdigit() else int(h_raw)
        if st is not None and 0 <= st <= 23:
            if st <= 11 and any(w in t for w in ["abend", "nachmittag", "pm"]):
                st += 12
            return st, 0
    # HH:MM
    m = re.search(r'\b(\d{1,2})[:\.](\d{2})\b', t)
    if m:
        st, mi = int(m.group(1)), int(m.group(2))
        if 0 <= st <= 23 and 0 <= mi <= 59:
            return st, mi
    return None


def parse_datum(text: str) -> Optional[date]:
    t = text.lower().strip()
    today = date.today()
    if "heute" in t:
        return today
    if "\u00fcbermorgen" in t or "übermorgen" in t:
        return today + timedelta(days=2)
    if "morgen" in t and "\u00fcbermorgen" not in t and "übermorgen" not in t:
        return today + timedelta(days=1)
    # in X tagen/wochen
    m = re.search(r'in\s+(\w+)\s+(tag|woche)', t)
    if m:
        n_raw = m.group(1)
        n_val = _WORT_ZU_ZAHL.get(n_raw) or (int(n_raw) if n_raw.isdigit() else None)
        if n_val:
            mult = 7 if 'woche' in m.group(2) else 1
            return today + timedelta(days=int(n_val * mult))
    # nächsten Wochentag
    m = re.search(r'n[\u00e4a]chsten?\s+(\w+)', t)
    if m:
        wt_name = m.group(1).capitalize()
        if wt_name in WOCHENTAGE_DE:
            wt_idx = WOCHENTAGE_DE.index(wt_name)
            delta = (wt_idx - today.weekday() + 7) % 7
            if delta == 0:
                delta = 7
            return today + timedelta(days=delta)
    # Wochentag allein
    for i, wt in enumerate(WOCHENTAGE_DE):
        if wt.lower() in t:
            delta = (i - today.weekday() + 7) % 7
            if delta == 0:
                delta = 7
            return today + timedelta(days=delta)
    # DD.MM.YYYY
    m = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})', t)
    if m:
        tag, mo, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if yr < 100:
            yr += 2000
        try:
            return date(yr, mo, tag)
        except ValueError:
            pass
    # DD.MM.
    m = re.search(r'(\d{1,2})\.(\d{1,2})\.?', t)
    if m:
        try:
            d = date(today.year, int(m.group(2)), int(m.group(1)))
            if d < today:
                d = date(today.year + 1, int(m.group(2)), int(m.group(1)))
            return d
        except ValueError:
            pass
    # DD. Monatsname
    m = re.search(r'(\d{1,2})\.\s*([a-z\u00e4\u00fc\u00f6]+)(?:\s+(\d{4}))?', t)
    if m:
        mo_name = m.group(2)
        mo = MONATE.get(mo_name)
        if not mo:
            for k, v in MONATE.items():
                if mo_name.startswith(k[:3]):
                    mo = v
                    break
        yr = int(m.group(3)) if m.group(3) else today.year
        if mo:
            try:
                d = date(yr, mo, int(m.group(1)))
                if d < today and not m.group(3):
                    d = date(today.year + 1, mo, int(m.group(1)))
                return d
            except ValueError:
                pass
    return None


def fmt_datum(d: date) -> str:
    return f"{WOCHENTAGE_DE[d.weekday()]}, {d.day}. {MONATE_DE[d.month]} {d.year}"


def parse_timer_sekunden(text: str) -> Optional[int]:
    t = text.lower()
    if "halbe stunde" in t or "halben stunde" in t:
        return 1800
    if "dreiviertel stunde" in t:
        return 2700
    if "viertel stunde" in t or "viertelstunde" in t:
        return 900
    t = _zahlen_normalisieren(t)
    total = 0.0
    found = False
    m = re.search(r'(\d+(?:\.\d+)?)\s*stunden?', t)
    if m:
        total += float(m.group(1)) * 3600; found = True
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:minuten?|min(?!\w))', t)
    if m:
        total += float(m.group(1)) * 60; found = True
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:sekunden?|sek(?!\w))', t)
    if m:
        total += float(m.group(1)); found = True
    if not found:
        m = re.search(r'\b(\d+)\b', t)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 120:
                total = n * 60; found = True
    return int(total) if found and total > 0 else None


class Kalender:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._d = {}
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._d = json.load(f)
            except Exception:
                pass

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._d, f, ensure_ascii=False, indent=2)

    def _k(self, d): return d.strftime("%Y-%m-%d")
    def get(self, d): return self._d.get(self._k(d), [])

    def add(self, d, text):
        k = self._k(d)
        if k not in self._d:
            self._d[k] = []
        if text.lower() not in [e.lower() for e in self._d[k]]:
            self._d[k].append(text)
            self._save()
            return True
        return False

    def remove(self, d, text=None):
        k = self._k(d)
        if k not in self._d:
            return 0
        b = len(self._d[k])
        if text:
            self._d[k] = [e for e in self._d[k] if text.lower() not in e.lower()]
        else:
            self._d[k] = []
        n = b - len(self._d[k])
        if not self._d[k]:
            del self._d[k]
        self._save()
        return n

    def upcoming(self, n=10):
        today = date.today()
        res = []
        for k in sorted(self._d.keys()):
            try:
                d = datetime.strptime(k, "%Y-%m-%d").date()
            except Exception:
                continue
            if d >= today:
                for e in self._d[k]:
                    res.append((d, e))
        return res[:n]


kalender = Kalender(CONFIG["db_path"])


class Notizen:
    """Notizen ohne Ablauf – bleiben bis explizit gelöscht."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._n = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._n = []
                for item in data:
                    if isinstance(item, dict):
                        self._n.append(item.get("text", ""))
                    else:
                        self._n.append(str(item))
            except Exception:
                self._n = []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._n, f, ensure_ascii=False, indent=2)

    def add(self, text: str):
        self._n.append(text)
        self._save()

    def get_all(self): return list(self._n)
    def clear(self):
        self._n = []
        self._save()

    def delete_matching(self, query: str) -> int:
        before = len(self._n)
        q = query.lower().strip()
        self._n = [n for n in self._n if q not in n.lower()]
        deleted = before - len(self._n)
        if deleted:
            self._save()
        return deleted


notizen = Notizen(CONFIG["notes_path"])


class TodoListe:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: list = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._items = json.load(f)
            except Exception:
                self._items = []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._items, f, ensure_ascii=False, indent=2)

    def add(self, text: str) -> bool:
        if text.lower() not in [i.lower() for i in self._items]:
            self._items.append(text)
            self._save()
            return True
        return False

    def remove(self, query: str) -> int:
        before = len(self._items)
        q = query.lower().strip()
        self._items = [i for i in self._items if q not in i.lower()]
        deleted = before - len(self._items)
        if deleted:
            self._save()
        return deleted

    def get_all(self) -> list: return list(self._items)
    def clear(self):
        self._items = []
        self._save()


todo_liste = TodoListe(CONFIG["todo_path"])


class ErinnerungsManager:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._erinnerungen: list = []
        self._loop = None
        self._broadcast_cb = None
        self._load()
        self._thread = threading.Thread(target=self._checker_loop,
                                        daemon=True, name="erinnerung-checker")
        self._thread.start()

    def set_loop(self, loop, broadcast_cb):
        self._loop = loop
        self._broadcast_cb = broadcast_cb

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._erinnerungen = json.load(f)
            except Exception:
                self._erinnerungen = []

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._erinnerungen, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"ErinnerungsManager save: {e}")

    def add(self, zeitpunkt: datetime, text: str) -> bool:
        self._erinnerungen.append({
            "zeit": zeitpunkt.isoformat(),
            "text": text,
            "ausgelöst": False,
        })
        self._save()
        return True

    def get_all(self) -> list:
        now = datetime.now()
        return [e for e in self._erinnerungen
                if not e.get("ausgelöst", False)
                and datetime.fromisoformat(e["zeit"]) > now]

    def clear(self):
        self._erinnerungen = []
        self._save()

    def _checker_loop(self):
        while True:
            try:
                now = datetime.now()
                triggered = []
                for e in self._erinnerungen:
                    if e.get("ausgelöst", False):
                        continue
                    try:
                        zeit = datetime.fromisoformat(e["zeit"])
                    except Exception:
                        continue
                    if abs((now - zeit).total_seconds()) <= 120 and now >= zeit:
                        triggered.append(e)
                        e["ausgelöst"] = True
                if triggered:
                    self._save()
                    for e in triggered:
                        log.info(f"Erinnerung ausgelöst: {e['text']}")
                        if self._loop and self._broadcast_cb:
                            asyncio.run_coroutine_threadsafe(
                                self._fire_erinnerung(e["text"]), self._loop
                            )
            except Exception as ex:
                log.error(f"ErinnerungsChecker: {ex}")
            time.sleep(30)

    async def _fire_erinnerung(self, text: str):
        msg = f"Erinnerung: {text}"
        try:
            wav = await asyncio.get_event_loop().run_in_executor(None, synthesize, msg)
            play_audio_windows(wav)
            if self._broadcast_cb:
                await self._broadcast_cb({
                    "type": "erinnerung_alarm",
                    "text": msg,
                    "tts_audio": base64.b64encode(wav).decode(),
                    "mime": "audio/wav",
                })
            # ── Auch an App-Audio-Out-Clients senden ────────────────────
            await app_audio_broadcast(wav, "tts")
        except Exception as e:
            log.error(f"Erinnerung TTS: {e}")


erinnerungs_manager = ErinnerungsManager(CONFIG["reminders_path"])


class TimerManager:
    """
    Persistenter Timer-Manager v2: speichert Timer/Wecker als absolute Zeitpunkte
    in einer JSON-Datei (data/timers.json). Ein Background-Checker prüft alle 2s
    ob ein Timer fällig ist. Übersteht damit Reconnects (Geräte verbinden alle 3h neu).
    Timer sind per-Owner (Client-IP oder Device-Group-Name) isoliert:
    Client x.x.x.1 hört NICHT den Timer von x.x.x.2 (es sei denn, sie sind
    in derselben Device-Group).
    """
    def __init__(self):
        self._path = Path(CONFIG.get("timers_path", "data/timers.json"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._timers: list = []   # [{id, fire_at, label, source_ip, owner_key, total_sek, next_reminder_at}]
        self._counter = 0
        self._alarm_active: set = set()        # set of owner_keys mit laufendem Alarm
        self._alarm_source_ips: dict = {}      # owner_key → source_ip (für Audio-Targeting)
        self._broadcast_cb = None
        self._loop = None
        self._checker_task = None
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Lade Timer aus JSON-Datei. Abgelaufene Timer werden entfernt."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._timers = data.get("timers", [])
                self._counter = data.get("counter", 0)
                # Abgelaufene Timer entfernen (z.B. wenn Server neu startet)
                now_str = datetime.now().isoformat()
                before = len(self._timers)
                self._timers = [t for t in self._timers if t["fire_at"] > now_str]
                if len(self._timers) < before:
                    log.info(f"Timer-Datei: {before - len(self._timers)} abgelaufene Timer entfernt.")
                self._save()
            except Exception as e:
                log.error(f"Timer-Datei laden fehlgeschlagen: {e}")
                self._timers = []
        log.info(f"TimerManager: {len(self._timers)} aktive Timer geladen.")

    def _save(self):
        """Speichere Timer-State in JSON-Datei."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({"timers": self._timers, "counter": self._counter},
                          f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"Timer-Datei speichern fehlgeschlagen: {e}")

    def _owner_key(self, source_ip: str) -> str:
        """Bestimmt den Owner-Key: Gruppenname (wenn in Gruppe) oder IP."""
        if source_ip:
            gname = device_group_mgr.get_group_name(source_ip)
            if gname:
                return f"group:{gname}"
            return f"ip:{source_ip}"
        return "local"

    def set_loop(self, loop, broadcast_cb):
        self._loop = loop
        self._broadcast_cb = broadcast_cb
        # Starte den Background-Checker
        self._checker_task = asyncio.run_coroutine_threadsafe(
            self._checker_loop(), self._loop
        )

    def starten(self, sekunden: int, label: str = "", source_ip: str = None) -> str:
        """Timer starten: speichert absolute Zielzeit in Datei."""
        with self._lock:
            self._counter += 1
            tid = f"t{self._counter}"
            fire_at = datetime.now() + timedelta(seconds=sekunden)
            owner = self._owner_key(source_ip)
            entry = {
                "id": tid,
                "fire_at": fire_at.isoformat(),
                "label": label,
                "source_ip": source_ip,
                "owner_key": owner,
                "total_sek": sekunden,
            }
            # Reminder für lange Timer (>20min): alle timer_reminder_interval Sekunden
            REMINDER_INTERVAL = CONFIG.get("timer_reminder_interval", 600)
            if sekunden > 1200:
                next_rem = datetime.now() + timedelta(seconds=REMINDER_INTERVAL)
                entry["next_reminder_at"] = next_rem.isoformat()
            self._timers.append(entry)
            self._save()
            log.info(f"Timer {tid} gespeichert: feuert um {fire_at.strftime('%d.%m. %H:%M:%S')} "
                     f"'{label}' (owner: {owner})")
            return tid

    async def _checker_loop(self):
        """Background-Checker: prüft alle 2s ob Timer fällig sind / Reminder nötig."""
        while True:
            try:
                await asyncio.sleep(2)
                now = datetime.now()
                now_str = now.isoformat()
                fällig = []
                reminder_needed = []
                with self._lock:
                    neue_liste = []
                    for t in self._timers:
                        if t["fire_at"] <= now_str:
                            fällig.append(dict(t))
                        else:
                            # Reminder prüfen
                            rem_at = t.get("next_reminder_at")
                            if rem_at and rem_at <= now_str:
                                fire_dt = datetime.fromisoformat(t["fire_at"])
                                rest_sek = max(0, (fire_dt - now).total_seconds())
                                if rest_sek > 60:
                                    reminder_needed.append(dict(t))
                                    REMINDER_INTERVAL = CONFIG.get("timer_reminder_interval", 600)
                                    t["next_reminder_at"] = (now + timedelta(seconds=REMINDER_INTERVAL)).isoformat()
                            neue_liste.append(t)
                    if fällig or reminder_needed:
                        self._timers = neue_liste
                        self._save()
                # Fällige Timer auslösen
                for t in fällig:
                    asyncio.ensure_future(self._fire_alarm(t))
                # Reminder ansagen
                for t in reminder_needed:
                    fire_dt = datetime.fromisoformat(t["fire_at"])
                    rest = max(0, int((fire_dt - now).total_seconds()))
                    asyncio.ensure_future(self._sage_restzeit(t, rest))
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error(f"Timer-Checker Fehler: {e}")

    async def _fire_alarm(self, timer_entry: dict):
        """Timer feuern: Alarm-Sound NUR an den Owner (Client/Gruppe)."""
        tid = timer_entry["id"]
        label = timer_entry.get("label", "")
        source_ip = timer_entry.get("source_ip")
        owner = timer_entry.get("owner_key", "local")

        log.info(f"Timer {tid} ausgelöst: '{label}' (owner: {owner})")
        self._alarm_active.add(owner)
        self._alarm_source_ips[owner] = source_ip

        # Windows Sound (lokal auf Server)
        alarm_path = CONFIG.get("alarm_sound", "wecker.wav")
        if Path(alarm_path).exists():
            play_wav_file_windows(alarm_path, loop=True)

        msg = label if label else "Timer abgelaufen!"
        # ── Alarm-Event an Frontends (Banner) ────────────────────────────
        alarm_event = {
            "type": "timer_alarm",
            "timer_id": tid,
            "label": msg,
        }
        if self._broadcast_cb:
            await self._broadcast_cb(alarm_event)
        await app_text_broadcast(alarm_event)

        # ── Alarm-Audio NUR an Owner's Clients (Loop mit Pausen) ─────────
        try:
            alarm_wav = generate_alarm_wav()
            loop_secs = CONFIG.get("alarm_loop_secs", 60)
            loop_dur = (len(alarm_wav) - 44) / (16000 * 2)
            pause_dur = 2.0  # 2s Stille → OWW kann Wakeword erkennen
            elapsed = 0.0
            while owner in self._alarm_active and elapsed < loop_secs:
                # Audio nur an Owner senden (Gruppe oder einzelne IP)
                if source_ip:
                    group_ips = device_group_mgr.get_group_ips(source_ip)
                    if group_ips:
                        for ip in group_ips:
                            await app_audio_broadcast(alarm_wav, "alarm", target_ip=ip)
                    else:
                        await app_audio_broadcast(alarm_wav, "alarm", target_ip=source_ip)
                else:
                    # Lokaler Timer (Browser) → an alle
                    await app_audio_broadcast(alarm_wav, "alarm", target_ip=None)
                await asyncio.sleep(max(loop_dur, 1.0))
                elapsed += loop_dur + pause_dur
                if owner in self._alarm_active:
                    await asyncio.sleep(pause_dur)
        except asyncio.CancelledError:
            pass
        except Exception as _e:
            log.error(f"Timer Alarm App-Audio: {_e}")
        # Cleanup
        self._alarm_active.discard(owner)
        self._alarm_source_ips.pop(owner, None)

    async def _sage_restzeit(self, timer_entry: dict, rest_sek: float):
        """Restzeit ansagen – nur an den Owner."""
        source_ip = timer_entry.get("source_ip")
        tid = timer_entry.get("id", "?")
        rest = int(rest_sek)
        mins = rest // 60
        sek = rest % 60
        if mins > 0 and sek > 0:
            text = f"Timer: noch {mins} Minuten und {sek} Sekunden."
        elif mins > 0:
            text = f"Timer: noch {mins} Minuten."
        else:
            text = f"Timer: noch {sek} Sekunden."
        try:
            loop = asyncio.get_event_loop()
            wav = await loop.run_in_executor(None, synthesize, text)
            play_audio_windows(wav)
            if self._broadcast_cb:
                await self._broadcast_cb({
                    "type": "timer_reminder",
                    "timer_id": tid,
                    "text": text,
                    "tts_audio": base64.b64encode(wav).decode(),
                    "mime": "audio/wav",
                })
            # ── Audio nur an den Owner ──────────────────────────────────
            if source_ip:
                group_ips = device_group_mgr.get_group_ips(source_ip)
                if group_ips:
                    for ip in group_ips:
                        await app_audio_broadcast(wav, "tts", target_ip=ip)
                else:
                    await app_audio_broadcast(wav, "tts", target_ip=source_ip)
            else:
                await app_audio_broadcast(wav, "tts", target_ip=None)
        except Exception as e:
            log.error(f"Timer Restzeit TTS: {e}")

    def stoppen(self, tid: str = None, source_ip: str = None):
        """Timer stoppen. Mit source_ip: nur Timer dieses Owners. Ohne: alle."""
        owner = self._owner_key(source_ip) if source_ip else None
        stop_windows_sound()
        with self._lock:
            if tid:
                # Einzelnen Timer entfernen
                self._timers = [t for t in self._timers if t["id"] != tid]
            elif owner:
                # Alle Timer dieses Owners entfernen
                self._timers = [t for t in self._timers if t["owner_key"] != owner]
            else:
                # Alle Timer entfernen (z.B. Shutdown)
                self._timers = []
            self._save()
        # Alarm deaktivieren
        if owner:
            self._alarm_active.discard(owner)
            self._alarm_source_ips.pop(owner, None)
        else:
            self._alarm_active.clear()
            self._alarm_source_ips.clear()

    def stoppe_alle(self, source_ip: str = None):
        """Alle Timer stoppen. Mit source_ip: nur Timer dieses Owners."""
        self.stoppen(None, source_ip)

    async def stoppen_und_benachrichtigen(self, tid: str = None, source_ip: str = None):
        """Stopp + Benachrichtigung an Owner's App-Clients (Audio-Out Stop + Text-WS)."""
        owner = self._owner_key(source_ip) if source_ip else None
        was_owner_active = (owner in self._alarm_active) if owner else bool(self._alarm_active)
        self.stoppen(tid, source_ip)
        if was_owner_active:
            # Audio-Out: explizites Stop-Signal → App hört sofort auf
            stop_msg = json.dumps({"type": "audio", "kind": "alarm_stop"})
            with _app_audio_out_clients_lock:
                targets = set(_app_audio_out_clients)
            # Nur an Owner's Clients senden
            target_ips = None
            if source_ip:
                group_ips = device_group_mgr.get_group_ips(source_ip)
                target_ips = group_ips if group_ips else {source_ip}
            for ws in targets:
                try:
                    if target_ips and ws.remote_address[0] not in target_ips:
                        continue
                    await ws.send(stop_msg)
                except Exception:
                    pass
            # Text-WS: Banner ausblenden
            await app_text_broadcast({"type": "alarm_cleared"})

    def aktive(self, source_ip: str = None) -> list:
        """Aktive Timer abfragen. Mit source_ip: nur Timer dieses Owners."""
        now = datetime.now()
        owner = self._owner_key(source_ip) if source_ip else None
        result = []
        with self._lock:
            for t in self._timers:
                if owner and t["owner_key"] != owner:
                    continue
                fire_at = datetime.fromisoformat(t["fire_at"])
                rest = max(0, int((fire_at - now).total_seconds()))
                result.append({
                    "id": t["id"],
                    "rest_sek": rest,
                    "label": t.get("label", ""),
                    "total_sek": t.get("total_sek", 0),
                    "owner_key": t["owner_key"],
                })
        return result

    @property
    def alarm_laeuft(self):
        """Global: läuft irgendein Alarm? (Für Wakeword-Threshold)"""
        return bool(self._alarm_active)

    def alarm_laeuft_fuer(self, source_ip: str) -> bool:
        """Läuft ein Alarm für diesen Owner?"""
        owner = self._owner_key(source_ip)
        return owner in self._alarm_active


timer_manager = TimerManager()


def hole_wetter() -> str:
    import urllib.request
    lat = CONFIG.get("weather_lat", 50.4833)
    lon = CONFIG.get("weather_lon", 8.2667)
    city = CONFIG.get("weather_city", "Weilburg")
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,weathercode,windspeed_10m,relativehumidity_2m"
        f"&daily=temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum"
        f"&timezone=Europe%2FBerlin&forecast_days=2"
    )
    WMO = {
        0:"klar", 1:"überwiegend klar", 2:"teilweise bewölkt", 3:"bewölkt",
        45:"neblig", 48:"gefrierender Nebel",
        51:"leichter Nieselregen", 53:"mäßiger Nieselregen", 55:"starker Nieselregen",
        61:"leichter Regen", 63:"mäßiger Regen", 65:"starker Regen",
        71:"leichter Schnee", 73:"mäßiger Schnee", 75:"starker Schnee",
        80:"Regenschauer", 81:"mäßige Regenschauer", 82:"starke Regenschauer",
        85:"Schneeschauer", 86:"starke Schneeschauer",
        95:"Gewitter", 96:"Gewitter mit Hagel", 99:"Gewitter mit starkem Hagel",
    }
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Sophie/3.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        curr = data.get("current", {})
        daily = data.get("daily", {})
        temp = curr.get("temperature_2m")
        wcode = curr.get("weathercode", 0)
        wind = curr.get("windspeed_10m")
        beschreibung = WMO.get(wcode, "unbekannt")
        temp_max = daily.get("temperature_2m_max", [None])[0]
        temp_min = daily.get("temperature_2m_min", [None])[0]
        niederschlag = daily.get("precipitation_sum", [None])[0]
        teile = []
        if temp is not None:
            teile.append(f"Aktuell {temp:.0f} Grad, {beschreibung}")
        if temp_min is not None and temp_max is not None:
            teile.append(f"Heute zwischen {temp_min:.0f} und {temp_max:.0f} Grad")
        if niederschlag is not None and niederschlag > 0:
            teile.append(f"{niederschlag:.1f} Millimeter Niederschlag erwartet")
        if wind is not None:
            teile.append(f"Wind {wind:.0f} Kilometer pro Stunde")
        if teile:
            return f"Wetter in {city}: " + ". ".join(teile) + "."
        return f"Wetterdaten für {city} nicht verfügbar."
    except Exception as e:
        log.error(f"Wetter API: {e}")
        return f"Ich konnte das Wetter für {city} gerade nicht abrufen."


def _oww_ensure_resources():
    import urllib.request
    try:
        import openwakeword
        res_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    except Exception as e:
        log.error(f"openwakeword nicht importierbar: {e}")
        return False
    res_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
    needed = ["melspectrogram.onnx", "embedding_model.onnx"]
    all_ok = True
    for fname in needed:
        dest = res_dir / fname
        if dest.exists():
            continue
        url = base_url + fname
        log.info(f"Lade openWakeWord Resource: {fname} ...")
        try:
            urllib.request.urlretrieve(url, str(dest))
            log.info(f"  OK {fname} bereit ({dest.stat().st_size // 1024} KB)")
        except Exception as e:
            log.error(f"  Fehler Download: {fname}: {e}")
            all_ok = False
    return all_ok


def get_cmd_model():
    global _cmd_model
    with _cmd_model_lock:
        if _cmd_model is None:
            import whisper
            log.info(f"Lade Command-Modell (whisper-{CONFIG['cmd_model']}) ...")
            _cmd_model = whisper.load_model(CONFIG["cmd_model"], device="cuda")
            log.info("Command-Modell bereit.")
        return _cmd_model



def get_tts(): return None

def setup_audio_device():
    """Listet Audio-Geräte und fragt den User nach der Auswahl."""
    print("\n" + "="*60)
    print(" AUDIO-OUTPUT SETUP")
    print("="*60)
    
    devices = sd.query_devices()
    valid_devices = []
    
    # Nur Ausgabegeräte anzeigen
    for idx, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            # Hostapi 0 ist meist MME/DirectSound (Standard Windows)
            print(f"  [{idx}] {device['name']} ({device['hostapi']})")
            valid_devices.append(idx)
            
    print("-" * 60)
    print("Drücke ENTER für Standard, oder gib die Nummer ein (z.B. für VB-Cable).")
    
    choice = input("Deine Wahl: ").strip()
    
    if choice and choice.isdigit() and int(choice) in valid_devices:
        dev_id = int(choice)
        CONFIG["audio_device_id"] = dev_id
        # ── WICHTIG: Global als Default setzen, sonst ignoriert Windows den Parameter ──
        try:
            current_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
            sd.default.device = (current_in, dev_id)
        except Exception:
            sd.default.device = dev_id  # Fallback: setzt beides
        name = devices[dev_id]['name']
        log.info(f"Audio-Output festgesetzt auf: [{dev_id}] {name}")
        print(f"  → Output: [{dev_id}] {name}")
        # Verify
        try:
            sd.check_output_settings(device=dev_id, samplerate=24000)
            print(f"  → Device OK ✓")
        except Exception as e:
            print(f"  → WARNUNG: Device-Check fehlgeschlagen: {e}")
    else:
        CONFIG["audio_device_id"] = None
        log.info("Nutze Windows Standard-Audioausgabe.")
    print("="*60 + "\n")

def play_audio_windows(wav_bytes: bytes):
    """Spielt TTS-Audio (Bytes) über das gewählte Device ab."""
    # ── Wakeword-Erkennung unterdrücken während Sophie lokal spricht ──
    global _wakeword_suppress_until
    try:
        # Dauer aus WAV-Header lesen (byte_rate @ offset 28)
        if len(wav_bytes) > 44 and wav_bytes[:4] == b'RIFF':
            byte_rate = struct.unpack('<I', wav_bytes[28:32])[0] or 32000
            pcm_bytes = max(0, len(wav_bytes) - 44)
            wav_dur = pcm_bytes / byte_rate
        else:
            wav_dur = 2.0  # Fallback
        _wakeword_suppress_until = max(_wakeword_suppress_until,
                                       time.time() + wav_dur + 0.5)
    except Exception:
        pass

    def _play():
        try:
            # Bytes in ein file-artiges Objekt umwandeln
            data, fs = sf.read(io.BytesIO(wav_bytes))
            # Abspielen (blocking=True innerhalb des Threads, damit es zu Ende spielt)
            sd.play(data, fs, device=CONFIG["audio_device_id"], blocking=True)
        except Exception as e:
            log.error(f"Audio Playback Error: {e}")

    # In Thread starten, damit Sophie nicht einfriert
    threading.Thread(target=_play, daemon=True, name="audio-playback").start()


_alarm_thread_running = False

def play_wav_file_windows(path: str, loop: bool = False):
    """Spielt eine WAV-Datei ab (z.B. Alarm), optional als Loop."""
    global _alarm_thread_running
    
    # Vorherigen Alarm stoppen, falls vorhanden
    stop_windows_sound()

    if not os.path.exists(path):
        log.error(f"Audio-Datei nicht gefunden: {path}")
        return

    def _play_loop():
        global _alarm_thread_running
        _alarm_thread_running = True
        try:
            data, fs = sf.read(path)
            # Loop-Logik
            while _alarm_thread_running:
                sd.play(data, fs, device=CONFIG["audio_device_id"], blocking=True)
                if not loop:
                    break
        except Exception as e:
            log.error(f"File Playback Error: {e}")
        finally:
            _alarm_thread_running = False

    threading.Thread(target=_play_loop, daemon=True, name="alarm-playback").start()


def stop_windows_sound():
    """Stoppt sofort alle Wiedergaben."""
    global _alarm_thread_running
    _alarm_thread_running = False
    try:
        sd.stop()
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
#  APP-MODE: Audio-Generierung für Chime/Beeps (server-seitig für Android)
# ═══════════════════════════════════════════════════════════════════════════════

import struct
import math as _math

def _generate_wav(samples_float: list, sample_rate: int = 16000) -> bytes:
    """Float-Samples [-1,1] → WAV-Bytes (16-bit PCM mono)."""
    n = len(samples_float)
    data = struct.pack(f'<{n}h', *[int(max(-32768, min(32767, s * 32767))) for s in samples_float])
    header = struct.pack('<4sI4s', b'RIFF', 36 + len(data), b'WAVE')
    fmt = struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    data_hdr = struct.pack('<4sI', b'data', len(data))
    return header + fmt + data_hdr + data


_chime_cache: Optional[bytes] = None
_beep_cache: Optional[bytes] = None


def generate_chime_wav() -> bytes:
    """Wakeword-Erkennungs-Chime als WAV."""
    global _chime_cache
    if _chime_cache:
        return _chime_cache
    sr = 16000
    dur = 0.45
    n = int(sr * dur)
    s = [0.0] * n
    for i in range(int(sr * 0.20)):
        t = i / sr
        env = min(1.0, t / 0.03) * max(0.001, _math.exp(-t / 0.10))
        s[i] += 0.28 * env * (2 * abs(2 * ((220 * t) % 1) - 1) - 1)
    for i in range(int(sr * 0.14), min(n, int(sr * 0.40))):
        t = (i - int(sr * 0.14)) / sr
        env = min(1.0, t / 0.03) * max(0.001, _math.exp(-t / 0.12))
        s[i] += 0.28 * env * (2 * abs(2 * ((330 * t) % 1) - 1) - 1)
    for i in range(int(sr * 0.32)):
        t = i / sr
        env = min(1.0, t / 0.04) * max(0.001, _math.exp(-t / 0.15))
        s[i] += 0.12 * env * _math.sin(2 * _math.pi * 110 * t)
    _chime_cache = _generate_wav(s, sr)
    return _chime_cache


def generate_thinking_beeps_wav() -> bytes:
    """Denk-Beeps (bup-bup ... bup-bup) als WAV – eine Sequenz ~2s."""
    global _beep_cache
    if _beep_cache:
        return _beep_cache
    sr = 16000
    dur = 2.0
    n = int(sr * dur)
    s = [0.0] * n
    def _bup(off_s, freq):
        st = int(sr * off_s)
        for i in range(int(sr * 0.15)):
            if st + i >= n:
                break
            t = i / sr
            env = min(1.0, t / 0.02) * max(0.001, _math.exp(-t / 0.06))
            s[st + i] += 0.08 * env * (2 * abs(2 * ((freq * t) % 1) - 1) - 1)
    _bup(0, 440)
    _bup(0.18, 440)
    _bup(1.0, 440)
    _bup(1.18, 660)
    _beep_cache = _generate_wav(s, sr)
    return _beep_cache


_alarm_cache: Optional[bytes] = None


def generate_alarm_wav() -> bytes:
    """Wecker-Alarm als WAV – Bass-Kick, HiHat, Melodie-Loop (~3s, zum Loopen geeignet)."""
    global _alarm_cache
    if _alarm_cache:
        return _alarm_cache
    import random as _rnd
    _rnd_state = _rnd.getstate()
    _rnd.seed(42)  # Deterministic noise

    sr = 16000
    bpm = 140
    beat = 60.0 / bpm          # ~0.43s pro Beat
    bars = 2                    # 2 Takte à 4 Beats = 8 Beats
    total = beat * 4 * bars     # ~3.43s
    n = int(sr * total)
    s = [0.0] * n

    # ── Bass-Kick (auf Beat 1 und 3 jedes Takts) ─────────────────────────
    kick_beats = [0, 2, 4, 6]
    for b in kick_beats:
        st = int(sr * beat * b)
        kick_len = int(sr * 0.15)
        for i in range(kick_len):
            if st + i >= n:
                break
            t = i / sr
            freq = 150 * _math.exp(-t * 15) + 50
            env = max(0.0, 1.0 - t / 0.15) ** 2
            s[st + i] += 0.35 * env * _math.sin(2 * _math.pi * freq * t)

    # ── HiHat (Noise-Burst auf jede Achtel) ───────────────────────────────
    eighth = beat / 2
    for h in range(int(4 * bars * 2)):  # 16 Achtel
        st = int(sr * eighth * h)
        hat_len = int(sr * 0.03)
        accent = 0.18 if h % 2 == 0 else 0.10
        for i in range(hat_len):
            if st + i >= n:
                break
            t = i / sr
            env = max(0.0, 1.0 - t / 0.03) ** 3
            noise = _rnd.uniform(-1, 1)
            s[st + i] += accent * env * noise

    # ── Melodie (catchy Alarm-Pattern) ────────────────────────────────────
    melody_notes = [
        (659, 0), (659, 1), (784, 2), (659, 3),    # Takt 1: E5 E5 G5 E5
        (880, 4), (784, 5), (659, 6), (0, 7),       # Takt 2: A5 G5 E5 rest
    ]
    for freq, b in melody_notes:
        if freq == 0:
            continue
        st = int(sr * beat * b)
        note_len = int(sr * beat * 0.7)
        for i in range(note_len):
            if st + i >= n:
                break
            t = i / sr
            dur_note = note_len / sr
            att = min(1.0, t / 0.008)
            rel = max(0.0, 1.0 - max(0.0, t - dur_note * 0.6) / (dur_note * 0.4))
            env = att * rel
            saw = 2 * ((freq * t) % 1) - 1
            sub = _math.sin(2 * _math.pi * (freq / 2) * t)
            s[st + i] += env * (0.18 * saw + 0.07 * sub)

    # ── Limiter ───────────────────────────────────────────────────────────
    peak = max(abs(v) for v in s) or 1.0
    if peak > 0.95:
        s = [v * 0.95 / peak for v in s]

    _rnd.setstate(_rnd_state)
    _alarm_cache = _generate_wav(s, sr)
    return _alarm_cache


# ═══════════════════════════════════════════════════════════════════════════════
#  APP-MODE: Client-Tracking für 8775 (Text-WS), 8776 (Audio-In), 8777 (Audio-Out)
# ═══════════════════════════════════════════════════════════════════════════════

_app_text_clients: Set = set()       # 8775 WS – empfangen Text-Updates
_app_text_clients_lock = threading.Lock()

_app_audio_out_clients: Set = set()  # 8777 WS – empfangen Audio
_app_audio_out_clients_lock = threading.Lock()

# ── Per-IP Tracking für App Audio-In (8776) ────────────────────────────────
# Verhindert Konflikte wenn ein Gerät reconnectet oder mehrere Geräte aktiv sind
_app_audio_in_states: dict = {}      # IP → ClientState
_app_audio_in_states_lock = threading.Lock()

# ── Globale Wakeword-Unterdrückung während TTS ──────────────────────────
# Zeitstempel bis wann OWW-Erkennung pausiert (Sophies eigene Stimme ignorieren)
_wakeword_suppress_until: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DEVICE GROUPS: Mehrere Geräte agieren als eines
# ═══════════════════════════════════════════════════════════════════════════════

class DeviceGroupManager:
    """
    Verwaltet Device-Gruppen: mehrere App-Clients (per IP) die als ein
    logisches Gerät funktionieren.
    
    Verhalten:
    - Alle Geräte einer Gruppe hören gleichzeitig auf das Wakeword
    - Erkennt ein Gerät das Keyword → dieses nimmt auf, die Peers werden gemutet
    - Thinking-Beeps gehen an ALLE Geräte der Gruppe
    - TTS wird an alle Geräte gesendet (2-Phasen-Sync: preload + play)
    """
    
    def __init__(self):
        self._groups: dict = {}           # group_name → set of IPs
        self._ip_to_group: dict = {}      # IP → group_name
        self._active_device: dict = {}    # group_name → IP des aktiven Geräts
        self._lock = threading.Lock()
    
    def load_config(self, groups_config: list):
        """Lädt Device-Groups aus der Config."""
        with self._lock:
            self._groups.clear()
            self._ip_to_group.clear()
            for g in groups_config:
                name = g.get("name", f"group_{len(self._groups)}")
                ips = set(g.get("ips", []))
                if len(ips) < 2:
                    log.warning(f"Device-Group '{name}' hat weniger als 2 IPs, überspringe.")
                    continue
                self._groups[name] = ips
                for ip in ips:
                    if ip in self._ip_to_group:
                        log.warning(f"IP {ip} ist in mehreren Gruppen! Nutze letzte: '{name}'")
                    self._ip_to_group[ip] = name
                log.info(f"Device-Group '{name}': {', '.join(sorted(ips))}")
    
    def get_group_name(self, ip: str) -> Optional[str]:
        """Gibt den Gruppennamen für eine IP zurück, oder None."""
        with self._lock:
            return self._ip_to_group.get(ip)
    
    def get_group_ips(self, ip: str) -> Optional[set]:
        """Gibt alle IPs der Gruppe zurück zu der diese IP gehört."""
        with self._lock:
            gname = self._ip_to_group.get(ip)
            if gname:
                return set(self._groups[gname])
            return None
    
    def get_peer_ips(self, ip: str) -> set:
        """Gibt die anderen IPs der Gruppe zurück (ohne die eigene)."""
        group_ips = self.get_group_ips(ip)
        if group_ips:
            return group_ips - {ip}
        return set()
    
    def is_in_group(self, ip: str) -> bool:
        """Prüft ob eine IP zu einer Device-Group gehört."""
        with self._lock:
            return ip in self._ip_to_group
    
    def set_active_device(self, ip: str) -> bool:
        """
        Markiert ein Gerät als aktiv (hat Wakeword erkannt, nimmt Befehl auf).
        Gibt True zurück wenn erfolgreich, False wenn bereits ein anderes aktiv ist.
        """
        with self._lock:
            gname = self._ip_to_group.get(ip)
            if not gname:
                return True  # Kein Gruppen-Gerät, immer OK
            current = self._active_device.get(gname)
            if current and current != ip:
                log.info(f"[DevGroup:{gname}] {ip} will aktiv werden, aber {current} ist bereits aktiv.")
                return False
            self._active_device[gname] = ip
            log.info(f"[DevGroup:{gname}] {ip} ist jetzt das aktive Gerät.")
            return True
    
    def clear_active_device(self, ip: str):
        """Gibt das aktive Gerät frei (Befehl fertig verarbeitet)."""
        with self._lock:
            gname = self._ip_to_group.get(ip)
            if gname and self._active_device.get(gname) == ip:
                del self._active_device[gname]
                log.info(f"[DevGroup:{gname}] {ip} ist nicht mehr aktiv.")
    
    def is_active_device(self, ip: str) -> bool:
        """Prüft ob dieses Gerät gerade das aktive ist."""
        with self._lock:
            gname = self._ip_to_group.get(ip)
            if not gname:
                return False
            return self._active_device.get(gname) == ip
    
    def has_active_device(self, ip: str) -> bool:
        """Prüft ob in der Gruppe dieses Geräts bereits eines aktiv ist."""
        with self._lock:
            gname = self._ip_to_group.get(ip)
            if not gname:
                return False
            return gname in self._active_device
    
    def get_active_for_group(self, ip: str) -> Optional[str]:
        """Gibt die IP des aktiven Geräts der Gruppe zurück."""
        with self._lock:
            gname = self._ip_to_group.get(ip)
            if gname:
                return self._active_device.get(gname)
            return None


device_group_mgr = DeviceGroupManager()


async def app_text_broadcast(msg: dict):
    """Sende Text-Event an alle App-WebView-Clients (8775)."""
    with _app_text_clients_lock:
        targets = set(_app_text_clients)
    data = json.dumps(msg)
    for ws in targets:
        try:
            await ws.send(data)
        except Exception:
            pass


def _normalize_wav_for_app(wav_bytes: bytes) -> bytes:
    """
    XTTS WAVs haben extra Chunks (LIST, fact) → PCM-Daten bei Offset 68+ statt 44.
    Konvertiert zu sauberem 44-Byte-Header WAV das Android problemlos lesen kann.
    """
    import wave as _wave_mod
    try:
        # soundfile ist am robustesten
        try:
            buf = io.BytesIO(wav_bytes)
            data, sr = sf.read(buf, dtype='float32')
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            return _generate_wav(data.tolist(), sample_rate=sr)
        except Exception:
            pass
        # Fallback: Python wave stdlib
        buf = io.BytesIO(wav_bytes)
        with _wave_mod.open(buf, 'rb') as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            sw = w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if sw == 2:
            pcm = np.frombuffer(frames, dtype=np.int16)
            if nch > 1: pcm = pcm.reshape(-1, nch).mean(axis=1).astype(np.int16)
            float_data = (pcm.astype(np.float32) / 32767.0).tolist()
        elif sw == 4:
            pcm = np.frombuffer(frames, dtype=np.int32)
            if nch > 1: pcm = pcm.reshape(-1, nch).mean(axis=1).astype(np.int32)
            float_data = (pcm.astype(np.float32) / 2147483647.0).tolist()
        else:
            return wav_bytes
        return _generate_wav(float_data, sample_rate=sr)
    except Exception as e:
        log.warning(f"WAV-Normalisierung fehlgeschlagen ({e}), sende Original")
        return wav_bytes


async def app_audio_broadcast(audio_bytes: bytes, kind: str = "tts", target_ip: str = None):
    """
    Sende Audio-Daten an App-Audio-Out-Clients (8777).
    target_ip=None → an ALLE Clients (z.B. Erinnerungen).
    target_ip="192.168.x.x" → NUR an Clients von dieser IP.
    """
    with _app_audio_out_clients_lock:
        targets = set(_app_audio_out_clients)
    if not targets:
        if target_ip:
            log.warning(f"App-Audio: Kein Audio-Out Client verbunden! (target: {target_ip})")
        return

    # TTS-WAVs normalisieren: XTTS gibt extra Chunks zurück → data-Offset != 44
    if kind == "tts":
        audio_bytes = _normalize_wav_for_app(audio_bytes)
        # ── Wakeword-Erkennung unterdrücken während Sophie spricht ────
        # WAV-Dauer schätzen + 0.5s Nachlauf (Raum-Echo)
        global _wakeword_suppress_until
        pcm_bytes = max(0, len(audio_bytes) - 44)
        wav_dur = pcm_bytes / (16000 * 2) if pcm_bytes > 0 else 1.0
        _wakeword_suppress_until = max(_wakeword_suppress_until,
                                       time.time() + wav_dur + 0.5)

    msg = json.dumps({
        "type": "audio",
        "kind": kind,
        "audio_b64": base64.b64encode(audio_bytes).decode(),
        "mime": "audio/wav",
    })
    sent = 0
    for ws in targets:
        try:
            # Per-IP Filter: nur an den anfragenden Client senden
            if target_ip and ws.remote_address[0] != target_ip:
                continue
            await ws.send(msg)
            sent += 1
        except Exception:
            pass
    if target_ip and sent == 0:
        connected_ips = {ws.remote_address[0] for ws in targets}
        log.warning(f"App-Audio: Kein Audio-Out von {target_ip}! "
                    f"Verbundene Audio-Out IPs: {connected_ips}")


async def app_audio_broadcast_group(audio_bytes: bytes, kind: str, source_ip: str):
    """
    Sende Audio an alle Geräte der Device-Group (oder nur an source_ip falls keine Gruppe).
    Für Chime, Beeps etc. – kein Sync nötig.
    """
    group_ips = device_group_mgr.get_group_ips(source_ip)
    if group_ips:
        for ip in group_ips:
            await app_audio_broadcast(audio_bytes, kind, target_ip=ip)
    else:
        await app_audio_broadcast(audio_bytes, kind, target_ip=source_ip)


async def app_audio_broadcast_group_synced(audio_bytes: bytes, source_ip: str):
    """
    Sende TTS-Audio synchron an alle Geräte der Device-Group.
    2-Phasen-Ansatz:
      1. 'tts_preload' → Audio an alle Geräte vorladen (noch nicht abspielen)
      2. Kurze Wartezeit für Netzwerk-Propagation
      3. 'tts_play' → Alle Geräte starten gleichzeitig
    
    Falls keine Gruppe: normales Senden.
    """
    group_ips = device_group_mgr.get_group_ips(source_ip)
    if not group_ips or len(group_ips) < 2:
        # Kein Sync nötig – normales Senden
        await app_audio_broadcast(audio_bytes, "tts", target_ip=source_ip)
        return
    
    # ── Phase 1: Audio vorladen auf allen Geräten ─────────────────────
    normalized = _normalize_wav_for_app(audio_bytes)
    
    # Wakeword-Unterdrückung für die Gruppe
    global _wakeword_suppress_until
    pcm_bytes = max(0, len(normalized) - 44)
    wav_dur = pcm_bytes / (16000 * 2) if pcm_bytes > 0 else 1.0
    _wakeword_suppress_until = max(_wakeword_suppress_until,
                                   time.time() + wav_dur + 0.5)
    
    preload_msg = json.dumps({
        "type": "audio",
        "kind": "tts_preload",
        "audio_b64": base64.b64encode(normalized).decode(),
        "mime": "audio/wav",
    })
    
    with _app_audio_out_clients_lock:
        targets = set(_app_audio_out_clients)
    
    # Sende preload an alle Gruppen-Geräte
    group_ws = []
    for ws in targets:
        if ws.remote_address[0] in group_ips:
            try:
                await ws.send(preload_msg)
                group_ws.append(ws)
            except Exception:
                pass
    
    log.info(f"[DevGroup] TTS preload an {len(group_ws)} Geräte gesendet.")
    
    # ── Phase 2: Kurze Wartezeit für Netzwerk-Propagation ─────────────
    # 150ms reicht für LAN – Audio-Daten müssen decodiert+gepuffert sein
    await asyncio.sleep(0.15)
    
    # ── Phase 3: Sync-Trigger an alle Geräte ──────────────────────────
    play_msg = json.dumps({"type": "audio", "kind": "tts_play"})
    for ws in group_ws:
        try:
            await ws.send(play_msg)
        except Exception:
            pass
    
    log.info(f"[DevGroup] TTS play-Trigger an {len(group_ws)} Geräte gesendet.")


def synthesize(text: str) -> bytes:
    import urllib.request
    url = CONFIG.get("tts_api_url", "http://localhost:8020/tts_to_audio/")
    speaker = CONFIG.get("tts_speaker", "1")
    language = CONFIG.get("tts_language", "de")
    payload = json.dumps({"text": text, "speaker_wav": speaker, "language": language}).encode()
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=70) as resp:
            wav = resp.read()
        if len(wav) < 100:
            raise ValueError(f"XTTS lieferte zu kurze Antwort ({len(wav)} bytes)")
        return wav
    except Exception as e:
        log.warning(f"XTTS-API nicht erreichbar ({e}), nutze pyttsx3 Fallback")
        return _synthesize_pyttsx3(text)


def _synthesize_pyttsx3(text: str) -> bytes:
    import pyttsx3
    eng = pyttsx3.init()
    eng.setProperty("rate", int(175 * CONFIG.get("tts_speed", 1.0)))
    voices = eng.getProperty("voices")
    for pref in ["hedda desktop", "hedda", "katja", "stefan"]:
        for v in voices:
            if pref in v.name.lower():
                eng.setProperty("voice", v.id)
                break
        else:
            continue
        break
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        eng.save_to_file(text, tmp)
        eng.runAndWait()
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try: os.unlink(tmp)
        except: pass





# ═══════════════════════════════════════════════════════════════════════════════
#  SOPHIE NLP v2  –  Sentence-Transformer Intent-Engine
#  Schicht 1: Regex-Vorfilter (Mikrosekunden, ~100 % Präzision)
#  Schicht 2: Sentence-Transformer Embedding-Matching (CUDA → CPU Fallback)
#  Schicht 3: Confidence-basierter LLM-Fallback
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re_nlp

# ── Gerät ermitteln: CUDA bevorzugt, sonst CPU ─────────────────────────────
def _get_device() -> str:
    try:
        import torch
        # WIR KOMMENTIEREN DAS HIER AUS:
        # if torch.cuda.is_available():
        #     log.info("NLP: CUDA verfügbar – nutze GPU für Sentence-Transformer.")
        #     return "cuda"
    except Exception:
        pass
    log.info("NLP: CPU-Modus erzwungen (Crash-Fix).")
    return "cpu"



# ── Schicht 1: Regex-Vorfilter ───────────────────────────────────────────────
# Jedes Tupel: (intent_name, compiled_regex)
# Sobald EINER der Patterns matcht → direktes Ergebnis, kein Modell nötig.

_REGEX_RULES: list = []

def _r(intent: str, *patterns: str):
    for p in patterns:
        _REGEX_RULES.append((intent, _re_nlp.compile(p, _re_nlp.IGNORECASE)))

# Uhrzeit / Datum
_r("uhrzeit",
   r'\bwie\s+sp[äa]t\b',
   r'\bwieviel\s+uhr\b', r'\bwie\s+viel\s+uhr\b',
   r'\buhrzeit\b', r'\bwie\s+spaet\b')
_r("datum_heute",
   r'\bwelches\s+datum\b', r'\bwelcher\s+tag\b',
   r'\bder\s+wievielte\b', r'\bwievielte\s+ist\b',
   r'\bdatum\s+heute\b', r'\bdatum\s+morgen\b',
   r'\bwelchen\s+wochentag\b', r'\bwelcher\s+wochentag\b')
# Wetter
_r("wetter",
   r'\bwetter\b', r'\btemperatur\b',
   r'\bwie\s+kalt\b', r'\bwie\s+warm\b',
   r'\bscheint\s+die\s+sonne\b')
# Wecker (Uhrzeit-basiert) – MUSS VOR timer_stellen stehen!
_r("wecker_stellen",
   # "wecker um X uhr" / "wecker auf halb X" / "wecker für halb X"
   r'\bwecker\s+(?:um|auf|f[üu]r)\s+(?:halb|viertel)\b',
   r'\bwecker\s+um\s+\w+(?:\s+uhr)?\b',
   r'\bwecker\s+auf\s+\w+\s+uhr\b',
   # "stell(e) (einen) wecker um X"
   r'\b(?:stell|setz)\w*\s+(?:(?:einen?|mir\s+einen?|dir\s+einen?)\s+)?wecker\s+um\s+',
   # "stell(e) (einen) wecker auf/für + Uhrzeit" (halb/viertel/X uhr, NICHT minuten/sekunden)
   r'\b(?:stell|setz)\w*\s+(?:(?:einen?|mir\s+einen?)\s+)?wecker\s+(?:auf|f[üu]r)\s+(?:halb|viertel)\b',
   r'\b(?:stell|setz)\w*\s+(?:(?:einen?|mir\s+einen?)\s+)?wecker\s+(?:auf|f[üu]r)\s+\w+\s+uhr\b',
   # "weck(e) mich um X uhr" / "weck mich morgen um X"
   r'\b(?:weck|wecke)\w*\s+mich\s+(?:morgen\s+)?um\b',
   # "wecker stellen um X"
   r'\bwecker\s+stellen\s+um\b',
   # "ich brauche einen wecker um" / "mach einen wecker um"
   r'\b(?:brauche?|mach)\w*\s+(?:einen?\s+)?wecker\s+um\b',
   # "wecker morgen (früh/abend) um X" / "wecker heute abend um X"
   r'\bwecker\s+(?:morgen|heute)\s+(?:fr[üu]h\s+|abend\s+|mittag\s+|nacht\s+)?um\b',
   # "wecker morgen um" (ohne fr[üu]h/abend)
   r'\bwecker\s+morgen\s+um\b')
# Timer (Dauer-basiert)
_r("timer_stellen",
   r'\b(?:stell|setz|starte?)\w*\s+(?:einen?\s+)?(?:timer|countdown)\b',
   # "stell einen wecker auf/für X minuten/sekunden" (Dauer, nicht Uhrzeit)
   r'\b(?:stell|setz)\w*\s+(?:einen?\s+)?wecker\s+(?:auf|f[üu]r)\s+\w+\s*(?:minuten?|sekunden?|stunden?)\b',
   r'\btimer\s+(?:auf|f[üu]r)\b',
   r'\b(?:weck|erinner)\w*\s+mich\s+in\b',
   r'\b\d+\s*(?:minuten?|sekunden?|stunden?)\s+timer\b')
_r("timer_stoppen",
   r'\b(?:timer|alarm|wecker|countdown)\s+(?:stopp\w*|abbr\w*|aus|beend\w*)\b',
   r'\b(?:stopp|stoppe|stop)\s+(?:den\s+)?(?:timer|alarm)\b',
   r'\b(?:halt[s]?\s+(?:die\s+)?klappe|halts?\s+maul|sei\s+still|ruhig)\b',
   r'\b(?:mach\s+(?:den\s+)?(?:alarm|piepser?|wecker|ton)\s+aus)\b',
   r'\b(?:h[öo]r\s+auf\s+(?:zu\s+)?piep\w*)\b',
   r'\b(?:alarm|piep\w*)\s+aufh[öo]ren\b')
_r("timer_abfragen",
   r'\bwie\s+lange\s+(?:l[äa]uft|hat|noch)\b',
   r'\btimer\s+(?:noch|status|restzeit)\b',
   r'\bnoch\s+wie\s+lange\b',
   r'\bwann\s+klingelt\b',
   r'\bwieviel\s+(?:zeit|minuten?)\s+noch\b')
# Kalender
_r("heute_termine",
   r'\b(?:was\s+(?:habe?\s+ich\s+)?heute|heute\s+(?:was|termine?|plan))\b',
   r'\bheutige[nr]?\s+termine?\b')
_r("nächste_termine",
   r'\b(?:kommende|n[äa]chste[n]?)\s+termine?\b',
   r'\balle\s+termine?\s+(?:zeigen|anzeigen)\b',
   r'\bterminen?[üu]bersicht\b')
_r("kalender_eintragen",
   r'\b(?:trag|f[üu]ge?|erstell|notiere?|mach)\w*\s+.*?(?:termin|eintrag|kalender)\b',
   r'\bneue[nr]?\s+(?:termin|kalendereintrag)\b',
   r'\bkalender\s+eintrag\w*\s+(?:erstell|anleg)\w*\b')
_r("kalender_löschen",
   r'\b(?:l[öo]sch|entfern|streich)\w*\s+.*?(?:termin|eintr[äa]ge?)\b',
   r'\btermin\s+(?:absagen?|l[öo]schen?|entfernen?)\b',
   # "lösche/entferne meinen Kalender" / "Kalender löschen/leeren"
   r'\b(?:l[öo]sch|entfern|leere?)\w*\s+.*?\bkalender\b',
   r'\bkalender\w*\s+(?:l[öo]schen?|leeren?|entfernen?)\b',
   # "kalendereinträge löschen" / "lösche kalendereinträge"
   # "lösche alle einträge" / "lösche die einträge" (auch ohne Kalender-Keyword)
   r'\b(?:l[öo]sch|entfern|streich)\w*\s+(?:(?:alle|die|den|das|meine?)\s+)?eintr[äa]ge?\b',
   # "meeting am donnerstag streichen" / "zahnarzt am freitag löschen" (Verb am Ende)
   r'.+\s+(?:am|vom)\s+\w+\s+(?:streichen|l[öo]schen|entfernen|absagen|canceln)\s*$')
_r("kalender_abfrage",
   r'\b(?:was\s+(?:habe?\s+ich|steht|ist|passiert)\s+am)\b',
   r'\b(?:termine?\s+(?:am|f[üu]r|an\s+dem))\b',
   r'\b(?:habe?\s+ich\s+(?:am|etwas\s+am))\b',
   r'\bwas\s+steht\s+.*?\bkalender\b',
   r'\bwas\s+steht\s+an\b')
# ── Todo  (WICHTIG: muss VOR Notizen stehen, damit "schreib auf todo-liste" ────
#           nicht als Notiz gefangen wird!)                                   ────
_r("todo_löschen",
   # spezifisch mit Item: "entferne X von der Todo-Liste"
   r'\b(?:entfern|l[öo]sch|streich)\w*\s+.{1,50}\s+von\s+(?:der|meiner)\s+(?:todo|aufgaben?)\s*-?\s*liste?\b',
   # alles löschen
   r'\btodo\s*-?\s*liste?\s+(?:leeren?|l[öo]schen?|entleeren?)\b',
   r'\balle\s+(?:todo\s+)?aufgaben?\s+(?:l[öo]schen?|entfernen?)\b',
   r'\baufgabe\s+erledigt\b',
   r'\babgehakt\b',
   # "lösche aufgabe X" / "aufgabe löschen"
   r'\b(?:l[öo]sch|entfern)\w*\s+(?:die\s+)?aufgabe\b',
   r'\baufgabe\s+(?:l[öo]schen?|entfernen?)\b')
_r("todo_hinzufuegen",
   r'\b(?:setz[e]?|trag\w*|f[üu]ge?\w*|schreib\w*)\s+(?:auf\s+(?:die|meine)|zur?)\s+todo\b',
   r'\btodo\s+(?:hinzuf[üu]gen?|erstellen?|anlegen?)\b',
   r'\bneue?\s+(?:todo\s+)?aufgabe\b',
   r'\baufgabe\s+hinzuf[üu]gen?\b',
   # "schreib auf die/meine To-Do-Liste"
   r'\b(?:schreib\w*|setz\w*|trag\w*)\s+(?:auf\s+)?(?:die|meine)\s+(?:to\s*-?\s*do)\s*-?\s*liste\b')
_r("todo_lesen",
   r'\bwas\s+steht\s+auf\s+(?:meiner|der)\s+todo\b',
   r'\bzeig\w*\s+(?:mir\s+)?(?:meine?\s+)?todo\b',
   r'\btodo\s*-?\s*liste?\s+(?:vorlesen|anzeigen)\b',
   r'\bmeine?\s+aufgaben?\b',
   r'\bwas\s+muss\s+ich\s+noch\s+machen\b',
   r'\bmeine?\s+todos?\b')

# ── Notizen  (WICHTIG: löschen VOR lesen, hinzufügen VOR lesen) ──────────────
# Schreiben / Hinzufügen  — muss vor notizen_lesen stehen!
_r("notiz_hinzufuegen",
   # "zu meinen Notizen hinzufügen" / "füge zu meinen Notizen hinzu"
   r'\b(?:zu\s+(?:mein\w+\s+)?notizen?\s+hinzuf[üu]gen?|notizen?\s+hinzuf[üu]gen?)\b',
   r'\bf[üu]ge?\w*\s+.{0,30}notizen?\s+hinzu\b',
   r'\bhinzuf[üu]gen\b.*\bnotiz\b',
   # "mach/erstell eine Notiz"
   r'\b(?:mach[e]?|erstell[e]?)\s+(?:eine?[n]?\s+)?notiz\b',
   r'\bneue?\s+notiz\b',
   # "notiere dass/mir/dir/das"
   r'\bnotiere?\s+(?:das|dass|mir|dir)\b',
   # "schreib auf / schreib mir auf" — aber NICHT wenn "todo" oder "liste" folgt
   r'\b(?:schreib\w*)\s+(?:dir\s+|mir\s+)?auf\b(?!.*\b(?:todo|to\s*-?\s*do|liste)\b)',
   # "merke dir / merk dir" — auch mit Wörtern dazwischen (z.B. "merk du dir")
   r'\bmerke?\w*\s+(?:\w+\s+)?dir\b',
   r'\bkannst\s+du\s+(?:das\s+)?notieren\b',
   r'\bvergiss\s+nicht\b')

# Löschen — muss VOR notizen_lesen stehen (sonst frisst "meine notizen" den Match)
_r("notiz_löschen",
   r'\b(?:l[öo]sch|entfern)\w*\s+(?:(?:alle?|meine?)\s+)?notizen?\b',
   r'\bnotizen?\s+(?:l[öo]schen?|leeren?|entfernen?|löschen)\b',
   r'\balle?\s+notizen?\s+(?:l[öo]schen?|entfernen?|leeren?)\b',
   # "lösch um alle meine Notizen" — "um" ist Whisper-Artefakt
   r'\bl[öo]sch\w*\s+\w*\s+(?:alle?\s+)?(?:meine?\s+)?notizen?\b',
   # "leere (alle/meine) Notizen" — nach ASR-Fix landet "lehre/lähre" hier
   r'\bleere?\w*\s+(?:(?:alle?|meine?)\s+)?notizen?\b',
   r'\bnotizen?\s+leeren?\b')

# Lesen — erst NACHDEM schreiben+löschen ausgeschlossen wurden
_r("notizen_lesen",
   r'\b(?:was\s+sind\s+meine|zeig(?:e?\s+mir)?\s+(?:meine?\s+)?)\s*notizen?\b',
   r'\b(?:alle\s+)?notizen?\s+(?:vorlesen|anzeigen|abfragen)\b',
   # "lies/les/liest mir alle Notizen vor" — auch nach ASR-Fix
   r'\b(?:lies|les|liest|lese)\w*\s+(?:mir\s+)?(?:alle?\s+)?(?:meine?\s+)?notizen?\b',
   r'\bnotizen?\s+vorlesen\b',
   r'\bwas\s+habe?\s+ich\s+(?:mir\s+)?notiert\b',
   r'\bwas\s+habe?\s+ich\s+mir\s+gemerkt\b',
   r'^\s*meine?\s+notizen?\s*$',
   r'\bzeig\w*\s+(?:mir\s+)?meine?\s+notizen?\b')

# ── Erinnerungen  (löschen VOR lesen) ────────────────────────────────────────
_r("erinnerung_löschen",
   r'\b(?:l[öo]sch|entfern)\w*\s+(?:(?:alle?|meine?)\s+)?erinnerungen?\b',
   r'\berinnerungen?\s+(?:l[öo]schen?|entfernen?|leeren?)\b')
_r("erinnerung_setzen",
   r'\berinnere?\s+mich\b',
   r'\b(?:setz[e]?|erstell[e]?|mach[e]?)\s+(?:eine?\s+)?erinnerung\b',
   r'\berinnerung\s+(?:f[üu]r|auf|setzen?)\b')
_r("erinnerungen_lesen",
   r'\b(?:welche|zeig\w*|alle\s+)\s*erinnerungen?\b',
   r'\bmeine?\s+erinnerungen?\b',
   r'\berinnerungen?\s+(?:anzeigen|zeigen|lesen)\b')

# Mathe
_r("mathe",
   r'\b(?:was\s+(?:ist|ergibt|macht)\s+)?\d+[\s,\.]*(?:plus|minus|mal|geteilt|durch|hoch|modulo)\b',
   r'\brechn[e]?\w*\s',
   r'\bberechn[e]?\w*\s',
   r'\bwie\s+viel\s+(?:ist|ergibt|macht)\s+\d',
   r'\bwurzel\s+(?:von|aus)\b',
   r'\bquadratwurzel\b',
   r'\b\d+\s+(?:plus|minus|mal|geteilt|durch|hoch)\s+\d',
   r'\bprozent\s+von\b',
   r'\b\d+\s*[\+\-\*\/\^]\s*\d',
   r'\bfakult[äa]t\b',
   r'\bsinus\b|\bcosinus\b|\btangens\b',
   r'\b(?:eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|zwölf)\s+(?:plus|minus|mal|geteilt|durch|hoch)\s+(?:eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|zwölf)\b')
# Witz
_r("witz",
   r'\bwitz\b', r'\bwitze\b',
   r'\bwas\s+lustiges\b',
   r'\bzum\s+lachen\b',
   r'\berz[äa]hl\w*\s+(?:mir\s+)?(?:einen?\s+)?witz\b',
   r'\bsag\w*\s+(?:mir\s+)?(?:einen?\s+)?witz\b',
   r'\bmach\w*\s+(?:mir\s+)?(?:einen?\s+)?witz\b')
# Hilfe
_r("hilfe",
   r'\b(?:was\s+kannst\s+du|hilfe|deine\s+funktionen|was\s+kann\s+sophie)\b',
   r'\bbefehlsübersicht\b', r'\bich\s+brauche\s+hilfe\b')

# ── Smalltalk Regex-Regeln ─────────────────────────────────────────────────
_r("smalltalk_identitaet",
   r'\bwer\s+bist\s+du\b', r'\bwie\s+hei[ßs]t\s+du\b',
   r'\bwas\s+bist\s+du\b', r'\bbist\s+du\s+(?:eine?\s+)?(?:ki|roboter|bot|mensch|computer)\b',
   r'\bhast\s+du\s+(?:einen?\s+)?namen\b',
   r'\bstell\s+dich\s+(?:mal\s+)?vor\b')
_r("smalltalk_ersteller",
   r'\bwer\s+hat\s+dich\s+(?:erstellt|gemacht|programmiert|gebaut|entwickelt|erschaffen|geschrieben)\b',
   r'\bwer\s+ist\s+dein\s+(?:ersteller|sch[öo]pfer|entwickler|vater|papa|creator|erfinder)\b',
   r'\bvon\s+wem\s+(?:wurdest|bist)\b',
   r'\bwer\s+steckt\s+hinter\s+dir\b',
   r'\bwoher\s+kommst\s+du\b')
_r("smalltalk_alter",
   r'\bwie\s+alt\s+bist\s+du\b',
   r'\bwann\s+wurdest\s+du\s+(?:geboren|erstellt|gemacht)\b',
   r'\bwie\s+lange\s+(?:gibt|existier)\w*\s+(?:es\s+)?dich\b',
   r'\bseit\s+wann\s+(?:gibt|bist)\b.*\bdich\b',
   r'\bhast\s+du\s+(?:ein\s+alter|einen?\s+geburtstag)\b')
_r("smalltalk_befinden",
   r'\bwie\s+geht\s*(?:[\u0027\u2019]?s?)\s*(?:dir)?\b',
   r'\bwie\s+gehts\b', r'\bwie\s+geht\s+es\s+dir\b',
   r'\balles\s+(?:klar|fit)\s+bei\s+dir\b',
   r'\bwie\s+f[üu]hlst\s+du\s+dich\b')
_r("smalltalk_danke",
   r'^danke\w*$', r'\bvielen\s+dank\b', r'\bbesten\s+dank\b',
   r'\bdanke\s+(?:dir|schön|sehr|sophie)\b',
   r'^merci$', r'^thanks$')


def _regex_classify(text: str):
    """Schicht 1: Regex-Schnellpfad. Gibt (intent, score) oder None zurück."""
    tl = text.lower()
    for intent, pattern in _REGEX_RULES:
        if pattern.search(tl):
            return intent, 1.0
    return None


# ── Schicht 2: Sentence-Transformer Engine ───────────────────────────────────

_st_model = None          # SentenceTransformer Instanz
_st_embeddings = None     # np.ndarray  [n_beispiele, dim]
_st_labels: list = []     # ["intent_name", ...]
_st_intent_idx: dict = {} # {"intent_name": [indices...]}
_st_lock = threading.Lock()
_st_device = "cpu"


def _build_intent_corpus() -> tuple[list[str], list[str]]:
    """Gibt (sätze, labels) für alle INTENTS zurück – inkl. normalisierter Form."""
    sätze, labels = [], []
    for intent in INTENTS:
        name = intent["name"]
        for ex in intent["beispiele"]:
            # Original und normalisierte Version beide aufnehmen → mehr Coverage
            for variant in {ex, _normalize_intent_light(ex)}:
                sätze.append(variant)
                labels.append(name)
    return sätze, labels


def _normalize_intent_light(text: str) -> str:
    """Leichte Normalisierung: Kleinschrift, Whitespace, häufige Höflichkeitsfloskeln."""
    t = text.lower().strip()
    for w in [" bitte", " kannst du", " konntest du", " doch",
              " kurz", " einfach", " eigentlich", " mal "]:
        t = t.replace(w, " ")
    # Umgangssprachliche Normalform
    replacements = [
        (r'\bstell[e]?\s+(?:mir\s+)?(?:einen?\s+)?timer\b', 'timer stellen'),
        (r'\bsetz[e]?\s+(?:einen?\s+)?timer\b',              'timer stellen'),
        (r'\bmach[e]?\s+(?:eine?\s+)?notiz\b',               'notiz erstellen'),
        (r'\bschreib\s+(?:dir\s+|mir\s+)?auf\b',             'notiz erstellen'),
        (r'\bmerke?\s+dir\b',                                 'notiz erstellen'),
        (r'\bwie\s+viel\s+uhr\b',                            'wie spaet'),
        (r'\bwieviel\s+uhr\b',                               'wie spaet'),
        (r'\btrag[e]?\s+(?:einen?\s+)?termin\s+ein\b',       'termin eintragen'),
    ]
    for pat, repl in replacements:
        t = re.sub(pat, repl, t)
    return re.sub(r'\s+', ' ', t).strip()


# Alias für Rückwärtskompatibilität mit process_command-Aufruf
def _normalize_intent(text: str) -> str:
    return _normalize_intent_light(text)


def get_intent_engine():
    """
    Lädt Sentence-Transformer (einmalig, thread-safe).
    Versucht zuerst CUDA, fällt auf CPU zurück.
    Gibt (model, embeddings, labels, intent_idx) zurück.
    """
    global _st_model, _st_embeddings, _st_labels, _st_intent_idx, _st_device

    with _st_lock:
        if _st_model is not None:
            return _st_model, _st_embeddings, _st_labels, _st_intent_idx

        device = _get_device()
        _st_device = device

        MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
        log.info(f"Lade Sentence-Transformer '{MODEL_NAME}' auf {device.upper()} ...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(MODEL_NAME, device=device)
            log.info(f"Sentence-Transformer bereit. ({device.upper()})")
        except Exception as e:
            log.error(f"Sentence-Transformer Ladefehler: {e}")
            log.warning("Fallback auf TF-IDF-Engine.")
            _st_model = "tfidf_fallback"   # Marker für Fallback
            return _st_model, None, None, None

        sätze, labels = _build_intent_corpus()
        log.info(f"Berechne Embeddings für {len(sätze)} Beispielsätze ...")
        embeddings = model.encode(
            sätze,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # kosinus = skalarprodukt bei L2-norm
        )
        log.info("Embeddings fertig.")

        # Index: intent_name → Liste von Zeilen-Indizes in embeddings
        idx_map: dict = {}
        for i, lbl in enumerate(labels):
            idx_map.setdefault(lbl, []).append(i)

        _st_model       = model
        _st_embeddings  = embeddings
        _st_labels      = labels
        _st_intent_idx  = idx_map
        return model, embeddings, labels, idx_map


def _st_classify(text: str, threshold: float = 0.50, top_k: int = 3) -> tuple:
    """
    Sentence-Transformer Klassifikation.
    Mittelwert der Top-K Similarity-Werte pro Intent.
    Gibt (intent_name, score) zurück.
    """
    model, embeddings, labels, idx_map = get_intent_engine()

    # TF-IDF-Fallback falls Modell nicht geladen werden konnte
    if model == "tfidf_fallback" or embeddings is None:
        return _tfidf_fallback_classify(text, threshold)

    import numpy as _np
    query_emb = model.encode(
        [text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]  # shape: (dim,)

    # Cosinus-Similarity = Skalarprodukt (wegen L2-Normalisierung)
    sims = embeddings @ query_emb  # shape: (n_sätze,)

    best_intent = "api_fallback"
    best_score  = 0.0
    for intent_name, indices in idx_map.items():
        intent_sims = sims[indices]
        k = min(top_k, len(intent_sims))
        # Top-K Werte mitteln
        top_vals = _np.partition(intent_sims, -k)[-k:]
        score = float(top_vals.mean())
        if score > best_score:
            best_score  = score
            best_intent = intent_name

    if best_score < threshold:
        log.info(f"ST Score zu niedrig ({best_intent} {best_score:.3f}) → API Fallback")
        return "api_fallback", best_score

    return best_intent, best_score


# ── TF-IDF Notfallback (falls sentence-transformers nicht installiert) ───────
_tfidf_vec   = None
_tfidf_imat  = None
_tfidf_lock  = threading.Lock()

def _tfidf_fallback_classify(text: str, threshold: float = 0.25) -> tuple:
    """Original TF-IDF Logik als Fallback."""
    global _tfidf_vec, _tfidf_imat
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

    with _tfidf_lock:
        if _tfidf_vec is None:
            all_ex = [_normalize_intent_light(ex)
                      for i in INTENTS for ex in i["beispiele"]]
            wvec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), sublinear_tf=True)
            cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            wmat = wvec.fit_transform(all_ex)
            cmat = cvec.fit_transform(all_ex)
            combined = sp.hstack([wmat * 2.0, cmat])
            intent_vecs = []
            idx = 0
            for intent in INTENTS:
                n = len(intent["beispiele"])
                mean_v = sp.csr_matrix(combined[idx:idx+n].mean(axis=0))
                intent_vecs.append(mean_v)
                idx += n
            _tfidf_vec  = (wvec, cvec)
            _tfidf_imat = sp.vstack(intent_vecs)

    wvec, cvec = _tfidf_vec
    normalized = _normalize_intent_light(text)
    wq   = wvec.transform([normalized])
    cq   = cvec.transform([normalized])
    q    = sp.hstack([wq * 2.0, cq])
    sims = _cos_sim(q, _tfidf_imat).flatten()
    idx  = int(np.argmax(sims))
    score = float(sims[idx])
    name  = INTENTS[idx]["name"]

    _LOCAL_KW = {
        "timer","alarm","wecker","notiz","notizen","erinnerung","erinnerungen",
        "kalender","termin","termine","eintrag","eintragen","trag",
        "uhrzeit","datum","wochentag","hilfe","stopp","stoppen","todo",
        "aufgabe","aufgaben","liste","wetter","temperatur","erinnere",
    }
    words = set(re.findall(r'\b\w+\b', normalized.lower()))
    has_kw = bool(words & _LOCAL_KW)
    if not has_kw and score < 0.55:
        return "api_fallback", score
    if score < threshold:
        return "api_fallback", score
    return name, score


# ═══════════════════════════════════════════════════════════════════════════════
#  FUZZY ASR-KORREKTUR  –  Schicht 1b
#  Whisper macht bei schlechtem Mikro / Hintergrundgeräuschen typische
#  Fehler: Lautersetzung, Silbenauslassung, Endungsverstümmelung.
#  Diese Schicht repariert den Text VOR Regex und ST.
# ═══════════════════════════════════════════════════════════════════════════════

def _levenshtein(a: str, b: str) -> int:
    """Schnelle Levenshtein-Distanz (keine externe Abhängigkeit)."""
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * lb
        for j, cb in enumerate(b):
            curr[j+1] = min(prev[j+1]+1, curr[j]+1,
                            prev[j] + (0 if ca == cb else 1))
        prev = curr
    return prev[lb]


# ── Phonetisches Normalisierungsdict ─────────────────────────────────────────
# Schlüssel = was Whisper liefern kann, Wert = kanonische Form
# Sortiert nach Länge (längere zuerst) damit spezifischere zuerst greifen.
_ASR_FIXES: list[tuple[str, str]] = sorted([
    # ── Verben / Aktionen ──────────────────────────────────────────────────
    # füge / füg / fügen (Whisper hört oft "függer", "fiege", "fühge", "föge")
    ("függer",       "füge"),
    ("fühger",       "füge"),
    ("fühge",        "füge"),
    ("föge",         "füge"),
    ("fiege",        "füge"),
    ("füge hinzu",   "füge hinzu"),   # korrekt, kein Fix nötig aber für Prio
    ("hinzufüger",   "hinzufügen"),
    ("hinzufüge",    "hinzufügen"),
    ("hinzufügt",    "hinzufügen"),
    # lösche / lösch (Whisper: "läscht", "leescht", "loscht", "lörsch")
    ("läscht",       "lösche"),
    ("leescht",      "lösche"),
    ("loscht",       "lösche"),
    ("löscht",       "lösche"),
    ("lörsch",       "lösche"),
    ("lörsche",      "lösche"),
    ("löschte",      "lösche"),
    ("lösch alle",   "lösche alle"),
    # leere / leeren (Whisper: "lehre", "lähre", "lähren", "leren")
    ("lähren",       "leeren"),
    ("lähre",        "leere"),
    ("lehren",       "leeren"),
    ("lehre",        "leere"),
    ("leren",        "leere"),
    ("lährt",        "leere"),
    # schreib / schreibe
    ("schreib mir",  "schreib mir"),
    ("schreibe mir", "schreib mir"),
    ("schreibe auf", "schreib auf"),
    ("schreibt",     "schreib"),
    # merke / merk
    ("merkst",       "merk"),
    ("merkste",      "merke"),
    ("merde",        "merke"),        # Whisper-Artefakt bei "merke"
    ("merk dir",     "merke dir"),
    # notiere / notier
    ("notiert",      "notiere"),
    ("notierst",     "notiere"),
    # stelle / stell
    ("stellt",       "stelle"),
    ("stellst",      "stelle"),
    # setze / setz
    ("setzt",        "setze"),
    ("setzst",       "setze"),
    # trage / trag
    ("trägt",        "trage"),
    ("trägst",       "trage"),
    # entferne
    ("entfernst",    "entferne"),
    ("entfernt",     "entferne"),
    # zeige / zeig
    ("zeigt",        "zeige"),
    ("zeigst",       "zeige"),
    # lies / les / lese (Whisper: "les", "lese", "leest")
    ("leest",        "lies"),
    ("lest",         "lies"),
    ("les mir",      "lies mir"),
    ("les alle",     "lies alle"),
    ("les die",      "lies die"),
    # ── Nomen ──────────────────────────────────────────────────────────────
    # notiz / notizen (Whisper: "ritzen", "nutzen", "notsen", "motzen")
    ("ritzen",       "notizen"),
    ("notsen",       "notizen"),
    ("motzen",       "notizen"),
    ("notzten",      "notizen"),
    ("notitsen",     "notizen"),
    ("nutzen",       "notizen"),      # Achtung: nur wenn kein anderer Kontext
    ("notis",        "notiz"),
    ("notiez",       "notiz"),
    # timer (Whisper: "taimer", "taimr", "teimer", "taima")
    ("taimer",       "timer"),
    ("taimr",        "timer"),
    ("teimer",       "timer"),
    ("tiemer",       "timer"),
    ("taima",        "timer"),
    # wecker (Whisper: "weker", "wekker", "wega", "wecka")
    ("weker",        "wecker"),
    ("wekker",       "wecker"),
    ("wecka",        "wecker"),
    ("wega",         "wecker"),
    # erinnerung (Whisper: "erinnerrung", "erinneruing")
    ("erinnerrung",  "erinnerung"),
    ("erinneruing",  "erinnerung"),
    ("erinnnerung",  "erinnerung"),
    # kalender (Whisper: "kalenter", "kalander")
    ("kalenter",     "kalender"),
    ("kalander",     "kalender"),
    # termin (Whisper: "termien", "termiin")
    ("termien",      "termin"),
    ("termiin",      "termin"),
    # wetter (Whisper: "weter", "weder")
    ("weder",        "wetter"),       # nur wenn Wetterkontext (sonst riskant)
    ("weter",        "wetter"),
    # aufgabe (Whisper: "aufgave", "aufgahe")
    ("aufgave",      "aufgabe"),
    ("aufgahe",      "aufgabe"),
    # todo (Whisper: "to do", "todu", "todoo")
    ("to do",        "todo"),
    ("todu",         "todo"),
    ("todoo",        "todo"),
    ("to-do",        "todo"),
    # datum (Whisper: "datun", "dattum")
    ("datun",        "datum"),
    ("dattum",       "datum"),
    # ── Füllwörter / Whisper-Artefakte ─────────────────────────────────────
    ("nämlich das",  "dass"),         # "nämlich das X" → "dass X" (Note-Inhalt)
    ("nämlich",      ""),             # Artefakt entfernen
    ("also das",     "dass"),
    ("lösch um ",    "lösche "),       # "lösch um alle" → "lösche alle" (Whisper-Artefakt)
    ("lösche um ",   "lösche "),       # ditto
    ("entferne um ", "entferne "),
    ("mal den",      ""),             # "leere mal den Notizen" → "leere Notizen"
    ("mal die",      ""),
    ("mal das",      ""),
    (" doch ",       " "),
    (" halt ",       " "),
], key=lambda x: -len(x[0]))


# ── Keyword-Fuzzy-Matching ────────────────────────────────────────────────────
# Wenn Levenshtein(wort, canonical) <= MAX_DIST → ersetzen
_FUZZY_KEYWORDS: dict[str, str] = {
    # Kanonisch → erlaubte Schreibweisen automatisch per Distanz
    "notiz":       "notiz",
    "notizen":     "notizen",
    "timer":       "timer",
    "kalender":    "kalender",
    "termin":      "termin",
    "erinnerung":  "erinnerung",
    "erinnerungen":"erinnerungen",
    "aufgabe":     "aufgabe",
    "aufgaben":    "aufgaben",
    "wetter":      "wetter",
    "wecker":      "wecker",
    "lösche":      "lösche",
    "leere":       "leere",
    "füge":        "füge",
    "hinzufügen":  "hinzufügen",
    "vorlesen":    "vorlesen",
    "anzeigen":    "anzeigen",
}
_FUZZY_MAX_DIST = 2   # max. 2 Editoperationen


def _fuzzy_correct_word(word: str) -> str:
    """
    Ersetzt ein einzelnes Wort durch das ähnlichste kanonische Keyword
    wenn Levenshtein-Distanz ≤ _FUZZY_MAX_DIST.
    Gibt das Originalwort zurück wenn kein guter Match.
    """
    if len(word) < 5:
        return word   # Kurze Wörter (fünf, vier, ...) NICHT anfassen → verhindert fünf→füge
    best_canonical = word
    best_dist = _FUZZY_MAX_DIST + 1
    for canonical in _FUZZY_KEYWORDS:
        # Längencheck als schneller Vorfilter
        if abs(len(word) - len(canonical)) > _FUZZY_MAX_DIST:
            continue
        d = _levenshtein(word, canonical)
        if d < best_dist:
            best_dist = d
            best_canonical = canonical
    return best_canonical if best_dist <= _FUZZY_MAX_DIST else word


def _asr_korrigiere(text: str) -> str:
    """
    Korrigiert typische Whisper-ASR-Fehler auf Deutsch.
    Schritt 1: String-Ersetzungen aus _ASR_FIXES (Phrasen + Wörter)
    Schritt 2: Wort-für-Wort Levenshtein gegen Keyword-Liste
    Gibt korrigierten Text zurück (lowercase).
    """
    t = text.lower().strip()
    original = t

    # ── Schritt 1: Phrase/Wort-Ersetzungen ───────────────────────────────
    for wrong, right in _ASR_FIXES:
        if wrong in t:
            t = t.replace(wrong, right)

    # ── Schritt 2: Levenshtein pro Wort auf Keyword-Liste ────────────────
    words = t.split()
    corrected = []
    for w in words:
        # Satzzeichen temporär abziehen
        punct = ""
        while w and w[-1] in ".,!?;:":
            punct = w[-1] + punct
            w = w[:-1]
        fixed = _fuzzy_correct_word(w)
        corrected.append(fixed + punct)
    t = " ".join(corrected)
    t = re.sub(r'\s+', ' ', t).strip()

    if t != original:
        log.info(f"ASR-Fix: '{original}' → '{t}'")
    return t


# ── Hauptfunktion: erkenneIntent ─────────────────────────────────────────────

def erkenneIntent(text: str) -> dict:
    """
    Vierstufige Intent-Erkennung:
      1a. ASR-Fuzzy-Korrektur  → Whisper-Fehler reparieren
      1b. Regex-Vorfilter      → sofort, kein Modell
      2.  Sentence-Transformer → Embedding-Matching (CUDA → CPU)
      3.  Confidence-Fallback  → api_fallback
    """
    tl = text.lower().strip()

    # ── Schicht 1a: ASR-Korrektur ─────────────────────────────────────────
    tl_fixed = _asr_korrigiere(tl)

    # ── Schicht 1b: Regex auf korrigiertem Text ───────────────────────────
    regex_result = _regex_classify(tl_fixed)
    if regex_result:
        intent_name, score = regex_result
        log.info(f"Regex-Intent: '{intent_name}' | '{text}'")
        return {"intent": intent_name, "score": score, "text": text}

    # ── Schicht 2: Sentence-Transformer ──────────────────────────────────
    # ST bekommt den korrigierten Text — bessere Embeddings
    normalized = _normalize_intent_light(tl_fixed)
    intent_name, score = _st_classify(normalized)
    log.info(f"ST-Intent: '{intent_name}' score={score:.3f} | '{text}' (fixed: '{tl_fixed}')")
    return {"intent": intent_name, "score": round(score, 3), "text": text}


def _extrahiere_beschreibung(text: str, d: Optional[date]) -> Optional[str]:
    l = text.lower().strip()
    if l.startswith("was ") or l.startswith("welche ") or l.startswith("zeige "):
        return None
    # --- Ende Neu ---
    m = re.search(r'\bdass\b\s+(?:ich\s+|dort\s+|er\s+|sie\s+|wir\s+)?(.+)$', l)
    if m:
        b = m.group(1).strip().rstrip('.,')
        for kw in ['eintragen', 'hinzufuegen', 'hinzu', 'eingetragen', 'merken', 'muss']:
            b = re.sub(r'\b' + kw + r'\b', '', b).strip()
        b = b.strip('., ')
        if len(b) > 2:
            return b
    if d:
        pats = [d.strftime('%d.%m.%Y'), d.strftime('%d.%m.'),
                f'{d.day}. {MONATE_DE.get(d.month, "")}',
                f'{d.day}.{d.month}.', f'{d.day}.']
        for pat in pats:
            idx = l.find(pat.lower())
            if idx >= 0:
                rest = l[idx + len(pat):].strip().strip(',:;')
                for kw in ['eintragen', 'hinzufuegen', 'hinzu', 'eintrag', 'eingetragen',
                           'termin', 'merken', 'notieren', 'kalender', 'mache', 'mach',
                           'einen', 'eine', 'der', 'die', 'das']:
                    rest = re.sub(r'\b' + kw + r'\b', '', rest, flags=re.IGNORECASE).strip()
                rest = re.sub(r'\s+', ' ', rest).strip('., ')
                if len(rest) > 2:
                    return rest
    prefixes_re = [
        r'trag(e)?\s+(mir\s+)?in\s+(den\s+)?kalender\s+ein[\s,]*(?:dass\s+)?',
        r'trag(e)?\s+ein[\s,]*(?:dass\s+)?',
        r'mach(e)?\s+einen?\s+eintrag[\s,]*(?:dass\s+)?',
        r'fuge?\s+(einen?\s+)?termin\s+ein[\s,]*',
        r'erstelle?\s+einen?\s+(?:kalender)?eintrag[\s,]*(?:dass\s+)?',
        r'schreib\s+in\s+(den\s+)?kalender[\s,]*(?:dass\s+)?',
        r'(neuer?\s+)?kalendereintrag[\s,]*',
        r'termin\s+eintragen[\s,]*(?:dass\s+)?',
    ]
    stripped = l
    for pat in prefixes_re:
        stripped = re.sub(pat, '', stripped, flags=re.IGNORECASE, count=1).strip()
    stripped = re.sub(r'\bfur\s+den\b', '', stripped, flags=re.IGNORECASE)
    stripped = re.sub(r'\bam\b', '', stripped, flags=re.IGNORECASE)
    stripped = re.sub(r'\b\d{1,2}\.\s*(?:\d{1,2}\.\s*(?:\d{2,4})?)?\b', '', stripped).strip()
    stripped = re.sub(
        r'\b(januar|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember'
        r'|jan|feb|mar|apr|jun|jul|aug|sep|okt|nov|dez|m\u00e4rz)\b',
        '', stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r'\b\d{4}\b', '', stripped).strip()
    for wt in WOCHENTAGE_DE:
        stripped = re.sub(r'\b' + wt.lower() + r'\b', '', stripped, flags=re.IGNORECASE)
    stripped = re.sub(r'\s+', ' ', stripped).strip('., ')
    if len(stripped) > 3:
        return stripped
    return None


def _extrahiere_notiz_inhalt(text: str) -> str:
    """
    Extrahiert nur den eigentlichen Notizinhalt aus einem Sprachbefehl.
    Beispiel: "mach eine Notiz dass ich morgen einkaufen muss"
              → "morgen einkaufen muss"
    Beispiel: "mache eine neue Notiz, nämlich, dass Flo ein Arschloch ist"
              → "Flo ein Arschloch ist"
    """
    # Reihenfolge ist wichtig: spezifischere Patterns zuerst
    prefixes = [
        # "mach/mache eine (neue) Notiz, nämlich (dass) ..."
        r'mach[e]?\s+(?:mir\s+)?(?:eine?[n]?|ne)\s+(?:neue?\s+)?notiz[,.\s]+(?:n[äa]mlich[,.\s]+)?(?:dass\s+)?(?:ich\s+)?',
        r'mache?\s+(?:mir\s+)?(?:eine?[n]?|ne)\s+(?:neue?\s+)?notiz[,.\s]+(?:n[äa]mlich[,.\s]+)?(?:dass\s+)?(?:ich\s+)?',
        # "füge eine neue Notiz hinzu, nämlich dass ..."
        r'f[üu]ge?\s+(?:eine?\s+)?(?:neue?\s+)?notiz\s+hinzu[,:\s]+(?:n[äa]mlich[,.\s]+)?(?:dass\s+)?(?:ich\s+)?',
        # "zu meinen Notizen hinzufügen, nämlich dass ..."
        r'zu\s+(?:mein\w+\s+)?notizen?\s+hinzuf[üu]gen?[,:\s]+(?:n[äa]mlich[,.\s]+)?(?:dass\s+)?(?:ich\s+)?',
        # "notiere dass ..." / "notier ..."
        r'notiere?\s+(?:bitte\s+)?(?:n[äa]mlich\s+)?(?:dass\s+)?(?:ich\s+)?',
        # "merke dir dass ..." / "merk dir ..."
        r'merk[e]?\s+dir\s+(?:bitte\s+)?(?:dass\s+)?(?:ich\s+)?',
        r'merk\s+(?:dir\s+)?(?:bitte\s+)?(?:dass\s+)?(?:ich\s+)?',
        # "schreib auf dass ..." / "schreib dir auf ..."
        r'schreib\s+(?:dir\s+|mir\s+)?auf[,\s]+(?:dass\s+)?(?:ich\s+)?',
        r'schreib\s+(?:dir\s+|mir\s+)?(?:auf\s+)?(?:dass\s+)?(?:ich\s+)?',
        # "erstelle eine Notiz ..."
        r'erstelle?\s+(?:eine?\s+)?(?:notiz|erinnerung)[,.\s]+(?:n[äa]mlich[,.\s]+)?(?:dass\s+)?(?:ich\s+)?',
        # "vergiss nicht dass ..."
        r'vergiss\s+nicht[,.\s]+(?:dass\s+)?(?:ich\s+)?',
        # "neue Notiz: ..."  /  "neue Notiz, nämlich ..."
        r'neue?\s+notiz[,:\s]+(?:n[äa]mlich[,.\s]+)?(?:dass\s+)?(?:ich\s+)?',
        # "kannst du (dir/das) notieren ..."
        r'kannst\s+du\s+(?:dir\s+|das\s+)?notieren[,\s]+(?:dass\s+)?(?:ich\s+)?',
    ]
    content = text.strip()
    for pat in prefixes:
        new_content = re.sub(pat, '', content, flags=re.IGNORECASE, count=1).strip()
        if new_content != content and len(new_content) >= 2:
            content = new_content
            break  # erstes passendes Pattern reicht

    # "nämlich" + kleine Füllwörter am Anfang entfernen
    content = re.sub(r'^(?:n[äa]mlich)[,.\s]+', '', content, flags=re.IGNORECASE).strip()
    # "dass/das" am Anfang nur entfernen wenn Verb folgt (nicht "das Brot...")
    content = re.sub(r'^dass\s+(?=ich\s|er\s|sie\s|wir\s|du\s)', '', content, flags=re.IGNORECASE)
    content = re.sub(r'^(?:mir|dir)\s+', '', content, flags=re.IGNORECASE)

    result = content.strip().rstrip('.,!').strip()

    # Plausibilitätscheck: ist genug echter Inhalt da?
    # Wenn nur Kommando-Wörter übrig sind → kein sinnvoller Notiz-Inhalt
    _COMMAND_ONLY = {'notiz', 'notizen', 'neue', 'neu', 'eine', 'einen', 'machen',
                     'erstellen', 'anlegen', 'hinzufügen', 'hinzu', 'bitte', 'mache',
                     'das', 'die', 'der', 'notiere', 'schreib', 'merke', 'merk'}
    words = set(re.findall(r'\b\w+\b', result.lower()))
    if len(result) < 3 or (words and words.issubset(_COMMAND_ONLY)):
        return ""
    return result


def _extrahiere_todo_inhalt(text: str) -> str:
    prefixes = [
        r'setze\s+auf\s+(meine|die)\s+todo\s+liste\s+dass\s+(ich\s+)?',
        r'setze\s+auf\s+(meine|die)\s+todo\s+liste\s+',
        r'fuge?\s+(zur?\s+)?(todo|aufgaben)\s*liste?\s+hinzu[,\s]+(?:dass\s+)?',
        r'trag\s+auf\s+die\s+todo\s+liste[,\s]+',
        r'schreib\s+auf\s+die\s+todo\s+liste[,\s]+',
        r'neue?\s+(todo\s+)?aufgabe[,\s:]+',
        r'aufgabe\s+hinzufuegen[,\s:]+',
        r'fuge?\s+(eine?\s+)?aufgabe\s+hinzu[,\s]+',
        r'todo\s+erstellen[,\s:]+',
    ]
    content = text.strip()
    for pat in prefixes:
        content = re.sub(pat, '', content, flags=re.IGNORECASE, count=1).strip()
    return content.strip().rstrip('.,').strip()


def _extrahiere_todo_löschen(text: str) -> Optional[str]:
    t = text.lower().strip()

    # Prüfe ob "alles löschen" gemeint ist
    alles_patterns = [
        r'\balle\b', r'\bkomplett\b', r'\bkomplette\b', r'\bgesamt\b',
        r'\bleeren?\b', r'\bcleared?\b',
    ]
    for pat in alles_patterns:
        if re.search(pat, t):
            return None

    # Entferne alle bekannten Kommando-/Füllwörter und schau was übrig bleibt
    generisch = re.sub(
        r'\b(todo|to|do|aufgabe[n]?|liste[n]?|entferne?[n]?|entfernst|löschen?|loschen?'
        r'|lösch[e]?|l\u00f6sch[e]?|streiche?|bitte|meine?|meiner|die|den|das|von'
        r'|der|aus|schei\w+|zeug|kram|stuff|eintr[äa]ge?|jetzt|bitte|tool'
        r'|du|ich|wir|sie|er|es|leeren?|leer|und|oder|erledigt|abgehakt)\b',
        '', t
    )
    generisch = re.sub(r'[.,!?\-]', '', generisch).strip()
    generisch = re.sub(r'\s+', ' ', generisch).strip()
    if len(generisch) < 3:
        # Kein spezifischer Inhalt erkannt → None = alles löschen
        return None

    # Spezifische Patterns für benanntes Item
    patterns = [
        r'entferne\s+(.+?)\s+von\s+der\s+todo\s*liste',
        r'lösch[e]?\s+(.+?)\s+von\s+der\s+todo\s*liste',
        r'l\u00f6sch[e]?\s+(.+?)\s+von\s+der\s+todo\s*liste',
        r'streiche?\s+(.+?)\s+(?:von|aus)\s+der\s+(?:todo\s*)?liste',
        r'(.+?)\s+ist\s+erledigt',
        r'(.+?)\s+abgehakt',
        r'entferne\s+(?:aufgabe\s+)?(.+)',
        r'lösch[e]?\s+(?:aufgabe\s+)?(.+)',
        r'l\u00f6sch[e]?\s+(?:aufgabe\s+)?(.+)',
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            result = m.group(1).strip()
            cleaned = re.sub(
                r'\b(bitte|die|den|das|meine?|von|der|aus|liste|todo|aufgabe'
                r'|entfernen|löschen|loschen)\b',
                '', result
            ).strip()
            cleaned = re.sub(r'[.,!?]', '', cleaned).strip()
            if len(cleaned) >= 2:
                return result.rstrip('.,!')
            return None

    # Letzter Fallback: nach Abzug aller Kommandowörter
    t2 = re.sub(r'^(entferne[n]?|entfernst|lösch[e]?|l\u00f6sch[e]?|streiche?|hak\s+ab)\s+', '', t).strip()
    t2 = re.sub(r'\s*(von|aus)\s*(der|meiner)\s*(todo\s*)?liste\s*$', '', t2).strip()
    t2 = re.sub(r'\s*bitte\s*', ' ', t2).strip()
    t2 = re.sub(r'\s*ist\s+erledigt\s*$', '', t2).strip()
    cleaned2 = re.sub(
        r'\b(bitte|die|den|das|meine?|aufgabe|todo|liste|entfernen|entfernst'
        r'|löschen|loschen|to|do|ist|erledigt|abgehakt|von|der|aus)\b',
        '', t2
    ).strip()
    cleaned2 = re.sub(r'[.,!?,\-]', '', cleaned2).strip()
    cleaned2 = re.sub(r'\s+', ' ', cleaned2).strip()
    if len(cleaned2) >= 2:
        return t2.rstrip('.,!')
    return None


def _extrahiere_notiz_löschen(text: str) -> Optional[str]:
    tl = text.lower()
    if re.search(r'\balle\b.*\bnotiz', tl) or re.search(r'\bnotiz.*\balle\b', tl):
        return None
    m = re.search(r'(?:lösch[e]?|l\u00f6sch[e]?|entferne?|streiche?)\s+(?:die\s+)?notiz\s+(?:uber\s+|\u00fcber\s+|von\s+|zu\s+)?(.+)', tl)
    if m:
        return m.group(1).strip().rstrip('.,')
    return None


def parse_erinnerung(text: str) -> Optional[tuple]:
    tl = text.lower().strip()
    now = datetime.now()
    # relative Zeit: "in X stunden/minuten"
    m = re.search(r'in\s+(\w+)\s+(stunden?|minuten?)', tl)
    if m:
        n_raw = m.group(1)
        einheit = m.group(2)
        n_val = _WORT_ZU_ZAHL.get(n_raw) or (int(n_raw) if n_raw.isdigit() else None)
        if n_val:
            delta_sek = int(n_val * (3600 if "stunde" in einheit else 60))
            zeitpunkt = now + timedelta(seconds=delta_sek)
            return zeitpunkt, _extrahiere_erinnerung_nachricht(tl)
    d = parse_datum(tl)
    uhrzeit = parse_uhrzeit(tl)
    if d and uhrzeit:
        stunde, minute = uhrzeit
        zeitpunkt = datetime(d.year, d.month, d.day, stunde, minute)
        if zeitpunkt < now:
            zeitpunkt += timedelta(days=1)
        return zeitpunkt, _extrahiere_erinnerung_nachricht(tl)
    if uhrzeit:
        stunde, minute = uhrzeit
        zeitpunkt = datetime(now.year, now.month, now.day, stunde, minute)
        if zeitpunkt < now:
            zeitpunkt += timedelta(days=1)
        return zeitpunkt, _extrahiere_erinnerung_nachricht(tl)
    return None


def _extrahiere_erinnerung_nachricht(text: str) -> str:
    tl = text.lower()
    m = re.search(r'\bdass\b\s+(?:ich\s+)?(.+?)(?:\s*$)', tl)
    if m:
        return m.group(1).strip().rstrip('.,')
    m = re.search(r'\ban\s+(?:den|die|das|meinen?\s+)?\s*(.+?)(?:\s*$)', tl)
    if m:
        return m.group(1).strip().rstrip('.,')
    txt = re.sub(r'erinnere?\s+mich\s+', '', tl, flags=re.IGNORECASE)
    txt = re.sub(r'\b(heute|morgen|\u00fcbermorgen|übermorgen)\b', '', txt)
    txt = re.sub(r'\bum\s+\S+\s+uhr\b', '', txt)
    txt = re.sub(r'\bin\s+\w+\s+(?:stunden?|minuten?)\b', '', txt)
    txt = re.sub(r'\s+', ' ', txt).strip().rstrip('.,')
    return txt if len(txt) > 2 else "Erinnerung"


def _bereinige_llm_output(raw: str) -> str:
    t = raw.strip()
    t = re.sub(r'^(Sophie|Assistent|Assistant|KI|AI|Bot)\s*:\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'[*_`#>|]+', '', t)
    t = re.sub(r'<[^>]+>', '', t)
    t = re.sub(r'[\r\n]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    saetze = re.split(r'(?<=[.!?])\s+', t)
    if len(saetze) > 3:
        t = ' '.join(saetze[:3])
    return t.strip('"').strip("'").strip()


def _chat_history_add(user_text: str, assistant_text: str):
    global _chat_history
    with _chat_history_lock:
        _chat_history.append((user_text, assistant_text))
        max_turns = CONFIG.get("api_history_turns", 3)
        if len(_chat_history) > max_turns:
            _chat_history = _chat_history[-max_turns:]


def api_anfrage(text: str) -> str:
    import urllib.request
    url = CONFIG.get("api_url", "")
    timeout = CONFIG.get("api_timeout", 90)
    max_len = CONFIG.get("api_max_length", 150)
    sysprompt = CONFIG.get("api_system_prompt", "")
    if not url:
        return "Ich habe keine API konfiguriert."
    prompt = "### Anweisung:\n" + sysprompt + "\n\n"
    with _chat_history_lock:
        history_snapshot = list(_chat_history)
    for user_msg, assistant_msg in history_snapshot:
        prompt += "### Benutzer:\n" + user_msg + "\n\n### Sophie:\n" + assistant_msg + "\n\n"
    prompt += "### Benutzer:\n" + text + "\n\n### Sophie:\n"
    payload = json.dumps({
        "prompt": prompt, "max_length": max_len, "max_context_length": 2048,
        "temperature": 0.65, "top_p": 0.9, "top_k": 40, "rep_pen": 1.1,
        "rep_pen_range": 256, "rep_pen_slope": 1, "tfs": 1, "top_a": 0,
        "typical": 1, "quiet": True, "trim_stop": True,
        "stop_sequence": ["### Benutzer:", "### Anweisung:", "### System:"],
    }).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data["results"][0]["text"]
        for stop in ["### Benutzer:", "### Anweisung:", "### System:"]:
            if stop in raw:
                raw = raw[:raw.index(stop)]
        result = _bereinige_llm_output(raw)
        if not result:
            return "Ich habe keine Antwort erhalten."
        _chat_history_add(text, result)
        return result
    except urllib.error.HTTPError as e:
        log.error(f"KoboldCPP HTTP {e.code}")
        return f"KoboldCPP meldet Fehler {e.code}."
    except urllib.error.URLError as e:
        log.error(f"KoboldCPP nicht erreichbar: {e.reason}")
        return "Moritz war wohl zu faul die KI anzuschalten."
    except Exception as e:
        log.error(f"KoboldCPP: {type(e).__name__}: {e}")
        return "Ich will gerade nicht antworten."


def _extrahiere_loesch_filter(text: str) -> Optional[str]:
    """
    Extrahiert einen spezifischen Suchbegriff für Kalender-Löschung.
    Gibt None zurück wenn ALLE Einträge gelöscht werden sollen (Standard!).
    Gibt einen Suchstring zurück NUR wenn ein ganz konkreter Termin-Name genannt wird
    wie z.B. "lösche den zahnarzt am freitag" → "zahnarzt"
    """
    tl = text.lower().strip()

    # Generische Löschung = ALLE löschen auf dem Datum:
    # "alle", "den/meinen kalender", "die einträge", "dass..." → alles löschen
    if re.search(r'\balle\b', tl):
        return None
    if re.search(r'\b(?:meinen?|den)\s+kalender\b', tl):
        return None
    if re.search(r'\bkalendereintr', tl):
        return None
    if re.search(r'\bdass\b', tl):
        return None  # "lösche den eintrag ... dass X" = beschreibend, nicht exakt

    # Generische "den eintrag" / "die einträge" ohne spezifischen Namen → alles
    if re.search(r'\b(?:den|die|das)\s+(?:eintr[äa]ge?|termine?|eintrag)\b', tl):
        return None

    # Versuche einen konkreten Termin-Namen zu extrahieren:
    # "lösche den zahnarzt am freitag" → "zahnarzt"
    # "meeting am donnerstag streichen" → "meeting"
    _GENERIC = {
        'eintrag', 'einträge', 'eintrage', 'termin', 'termine', 'kalender',
        'kalendereinträge', 'meinen', 'meine', 'meinem', 'den', 'die', 'das',
        'bitte', 'alle', 'am', 'vom', 'für', 'im', 'an', 'dem', 'der',
        'lösche', 'lösch', 'löschen', 'entferne', 'entfernen', 'streiche',
        'streichen', 'cancel', 'absagen', 'heute', 'morgen', 'übermorgen',
    }
    for wt in WOCHENTAGE_DE:
        _GENERIC.add(wt.lower())

    # Pattern: "lösche den ITEM am DATUM"
    m = re.search(r'\b(?:l[öo]sch|entfern|streich)\w*\s+(?:den\s+|die\s+|das\s+)?(.+?)\s+(?:am|vom|für|den)\s+', tl)
    if m:
        candidate = m.group(1).strip()
        words = set(re.findall(r'\b\w+\b', candidate))
        meaningful = words - _GENERIC
        # Entferne auch Datumsformate
        candidate_clean = re.sub(r'\d{1,2}\.\d{1,2}\.?\d{0,4}', '', candidate).strip()
        meaningful_words = set(re.findall(r'\b\w+\b', candidate_clean)) - _GENERIC
        if meaningful_words and len(' '.join(meaningful_words)) >= 3:
            return ' '.join(meaningful_words)

    # "meeting am donnerstag streichen" Pattern
    m = re.search(r'^(.+?)\s+(?:am|vom|für)\s+', tl)
    if m:
        candidate = m.group(1).strip()
        words = set(re.findall(r'\b\w+\b', candidate)) - _GENERIC
        if words and len(' '.join(words)) >= 3:
            return ' '.join(words)

    # Kein spezifischer Name erkannt → alles auf dem Datum löschen
    return None


import math as _math_module

_MATHE_WITZE = [
    "Wow, dafür brauchst du mich? Hat dein Taschenrechner Urlaub?",
    "Echt jetzt? Das hättest du auch im Kopf schaffen können. Naja, vielleicht auch nicht.",
    "Ich bin ein hochkomplexes KI-System und du lässt mich Grundschulmathe rechnen. Toll.",
    "Dafür wurde ich also erschaffen. Um dir beim Rechnen zu helfen. Mein Dasein hat endlich Sinn.",
    "Kopfrechnen ist wohl nicht so dein Ding, was? Kein Problem, dafür bin ich ja da.",
    "Hättest du in der Schule aufgepasst, müsstest du mich jetzt nicht fragen. Aber gut.",
    "Du weißt schon, dass dein Handy einen Taschenrechner hat, oder? Egal, hier kommt's:",
    "Mathe-Genie bist du offensichtlich keins. Aber kein Stress, ich schon.",
    "Ich bin quasi dein persönlicher Taschenrechner mit Persönlichkeit. Ob dir die gefällt, ist eine andere Frage.",
    "Oh, Mathe! Endlich mal was, wo ich besser bin als du. Wobei, das ist bei allem so.",
]


def _mathe_auswerten(text: str) -> Optional[str]:
    t = text.lower().strip()
    t = _zahlen_normalisieren(t)
    replacements = [
        (r'\bplus\b', '+'), (r'\bminus\b', '-'), (r'\bmal\b', '*'),
        (r'\bgeteilt\s+durch\b', '/'), (r'\bdurch\b', '/'),
        (r'\bgeteilt\b', '/'), (r'\bhoch\b', '**'),
        (r'\bmodulo\b', '%'), (r'\bmod\b', '%'),
    ]
    for pat, repl in replacements:
        t = re.sub(pat, repl, t)

    m = re.search(r'(?:quadrat)?wurzel\s+(?:von|aus)\s+(\d+(?:\.\d+)?)', t)
    if m:
        try: return f"{_math_module.sqrt(float(m.group(1))):g}"
        except: pass

    m = re.search(r'(\d+)\s*fakult', t) or re.search(r'fakult\w*\s+(?:von\s+)?(\d+)', t)
    if m:
        try:
            n = int(m.group(1))
            if n > 170: return "Das ist so groß, da bekommt selbst mein Prozessor Schweißausbrüche."
            result = _math_module.factorial(n)
            return f"{result:g}" if result < 1e15 else f"{result}"
        except: pass

    for fn, func in [("sinus", _math_module.sin), ("cosinus", _math_module.cos), ("tangens", _math_module.tan)]:
        m = re.search(rf'{fn}\s+(?:von\s+)?(\d+(?:\.\d+)?)', t)
        if m:
            try: return f"{func(_math_module.radians(float(m.group(1)))):.6g}"
            except: pass

    m = re.search(r'(\d+(?:\.\d+)?)\s*prozent\s+von\s+(\d+(?:\.\d+)?)', t)
    if m:
        try: return f"{(float(m.group(1)) / 100) * float(m.group(2)):g}"
        except: pass

    bruch_map = {"halb": 2, "drittel": 3, "viertel": 4, "fünftel": 5, "sechstel": 6,
                 "siebtel": 7, "achtel": 8, "neuntel": 9, "zehntel": 10}
    for bruch_name, divisor in bruch_map.items():
        m = re.search(rf'(?:ein\w?\s+)?{bruch_name}\s+von\s+(\d+(?:\.\d+)?)', t)
        if m:
            try: return f"{float(m.group(1)) / divisor:g}"
            except: pass

    expr = t
    expr = re.sub(r'^.*?(?:was\s+(?:ist|ergibt|macht)|rechne?\w*\s+(?:(?:mal\s+)?aus\s+)?(?:was\s+)?|berechne?\w*|wie\s+viel\s+(?:ist|ergibt|macht))\s*', '', expr)
    expr = re.sub(r'\s*(?:aus|ergibt|ist|bitte|denn)\s*$', '', expr).strip()
    expr = re.sub(r'[^0-9+\-*/().%\s\^]', ' ', expr)
    expr = re.sub(r'\s+', ' ', expr).strip()
    expr = expr.replace('^', '**')
    if not expr or not re.search(r'\d', expr): return None
    if not re.match(r'^[\d+\-*/().%\s\*]+$', expr): return None
    try:
        for f in ['import', 'exec', 'eval', 'open', 'os', 'sys', '__']:
            if f in expr: return None
        result = eval(expr, {"__builtins__": {}}, {})
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15: return f"{int(result)}"
            return f"{result:g}"
        return f"{result}"
    except ZeroDivisionError:
        return "Durch Null teilen? Ernsthaft? Das geht nicht mal in meiner Welt."
    except: return None


def handle_intent(r: dict) -> str:
    intent = r["intent"]
    text   = r["text"]

    # ── Kalender ──────────────────────────────────────────────────────
    if intent == "kalender_abfrage":
        d = parse_datum(text)
        if not d:
            return "Geb doch bitte ein Datum mit an du Idiot."
        t = kalender.get(d)
        if t:
            return f"Am {fmt_datum(d)} hast du: {', '.join(t)}."
        return f"Am {fmt_datum(d)} kann ich da nichts finden."

    elif intent == "kalender_eintragen":
        d = parse_datum(text)
        if not d:
            return "Ich konnte kein Datum erkennen. Bitte nenn mir Tag und Monat."
        b = _extrahiere_beschreibung(text, d)
        if not b or len(b) < 3:
            return f"Was soll ich für den {fmt_datum(d)} eintragen?"
        if kalender.add(d, b):
            return f"Eingetragen: {b} am {fmt_datum(d)}."
        return f"Das ist am {fmt_datum(d)} bereits eingetragen."

    elif intent == "kalender_löschen":
        d = parse_datum(text)
        if not d:
            # Kein Datum erkannt → versuche heute als Fallback
            # "lösch meinen kalender" ohne Datum → heute
            if re.search(r'\b(?:meinen?|den)\s+kalender\b', text.lower()):
                d = date.today()
            else:
                return "Ich konnte kein Datum erkennen. Für welchen Tag soll ich löschen?"
        b = _extrahiere_loesch_filter(text)
        n = kalender.remove(d, b)
        if n:
            if b:
                return f"{n} Eintrag mit '{b}' am {fmt_datum(d)} gelöscht."
            return f"{n} {'Eintrag' if n == 1 else 'Einträge'} am {fmt_datum(d)} gelöscht."
        return f"Nichts zum Löschen am {fmt_datum(d)} gefunden."

    elif intent == "heute_termine":
        t = kalender.get(date.today())
        if t:
            return "Deine heutigen Termine: " + ", ".join(t) + "."
        return "Du hast heute keine Termine. Also wirklich gar keine. Mach was draus, du fauler Sack."

    elif intent == "nächste_termine":
        u = kalender.upcoming(8)
        if u:
            return "Deine nächsten Termine: " + " - ".join(
                f"{fmt_datum(d)}: {e}" for d, e in u) + "."
        return "Keine zukünftigen Termine gefunden."

    # ── Notizen ───────────────────────────────────────────────────────
    elif intent == "notiz_hinzufuegen":
        content = _extrahiere_notiz_inhalt(text)
        if not content or len(content) < 3:
            return "Was soll ich notieren?"
        notizen.add(content)
        return f"Notiert: {content}."

    elif intent == "notizen_lesen":
        alle = notizen.get_all()
        if not alle:
            return "Du hast aktuell keine Notizen."
        if len(alle) == 1:
            return f"Du hast eine Notiz: {alle[0]}."
        aufzaehlung = ". ".join(f"{i+1}: {n}" for i, n in enumerate(alle))
        return f"Du hast {len(alle)} Notizen: {aufzaehlung}."

    elif intent == "notiz_löschen":
        query = _extrahiere_notiz_löschen(text)
        if query is None:
            notizen.clear()
            return "Alle Notizen wurden gelöscht."
        else:
            n = notizen.delete_matching(query)
            if n:
                return f"{n} Notiz gelöscht."
            return f"Keine Notiz mit '{query}' gefunden."

    # ── Timer ─────────────────────────────────────────────────────────
    elif intent == "timer_stellen":
        sek = parse_timer_sekunden(text)
        if not sek:
            return "Du Vollidiot, wie lange soll der Timer denn laufen? Sag doch einfach 'stell einen Timer auf 5 Minuten' oder so."
        mins  = sek // 60
        reste = sek % 60
        if mins > 0 and reste > 0:
            dauer = f"{mins} Minuten und {reste} Sekunden"
        elif mins > 0:
            dauer = f"{mins} Minuten"
        else:
            dauer = f"{sek} Sekunden"
        source_ip = r.get("source_ip")
        timer_manager.starten(sek, "", source_ip=source_ip)
        if sek > 1200:
            return f"Timer gestellt auf {dauer}."
        return f"Timer gestellt auf {dauer}."

    elif intent == "timer_stoppen":
        source_ip = r.get("source_ip")
        if timer_manager.alarm_laeuft_fuer(source_ip) if source_ip else timer_manager.alarm_laeuft:
            timer_manager.stoppe_alle(source_ip)
            return "Timer gestoppt."
        if timer_manager.aktive(source_ip):
            timer_manager.stoppe_alle(source_ip)
            return "Timer gestoppt."
        return "Es läuft gerade kein Timer."

    elif intent == "timer_abfragen":
        source_ip = r.get("source_ip")
        aktive = timer_manager.aktive(source_ip)
        if not aktive:
            return "Welcher verkorkste Timer? Es läuft gerade keiner."
        teile = []
        for t in aktive:
            rest = t["rest_sek"]
            mins = rest // 60
            sek = rest % 60
            if mins > 0 and sek > 0:
                teile.append(f"noch {mins} Minuten und {sek} Sekunden")
            elif mins > 0:
                teile.append(f"noch {mins} Minuten")
            else:
                teile.append(f"noch {sek} Sekunden")
        if len(teile) == 1:
            return f"Der Timer läuft {teile[0]}."
        return "Deine Timer: " + ". ".join(teile) + "."

    # ── Wecker (Uhrzeit-basiert) ──────────────────────────────────────
    elif intent == "wecker_stellen":
        zeit = parse_uhrzeit(text)
        if not zeit:
            return "Um wieviel Uhr soll der Wecker klingeln? Sag zum Beispiel 'Wecker um sieben Uhr dreißig'."
        stunde, minute = zeit
        now = datetime.now()
        # Ziel-Zeitpunkt bestimmen
        is_morgen = "morgen" in text.lower()
        ziel = now.replace(hour=stunde, minute=minute, second=0, microsecond=0)
        # Wenn Uhrzeit schon vorbei (oder "morgen" explizit): nächster Tag
        if ziel <= now or is_morgen:
            ziel += timedelta(days=1)
        delta_sek = int((ziel - now).total_seconds())
        if delta_sek <= 0:
            return "Die Uhrzeit liegt in der Vergangenheit. Versuch es nochmal."
        # Formatierung
        if minute == 0:
            zeit_str = f"{stunde} Uhr"
        else:
            zeit_str = f"{stunde} Uhr {minute:02d}"
        # Benutze TimerManager – gleicher Alarm-Sound + Stopp-Mechanismus
        source_ip = r.get("source_ip")
        timer_manager.starten(delta_sek, f"Wecker {zeit_str}", source_ip=source_ip)
        if is_morgen or ziel.date() != now.date():
            tag_str = "morgen" if (ziel.date() - now.date()).days == 1 else fmt_datum(ziel.date())
            return f"Wecker gestellt auf {tag_str} um {zeit_str}."
        return f"Wecker gestellt auf {zeit_str}."

    # ── Uhrzeit / Datum  (IMMER aus System, NIE LLM) ──────────────────
    elif intent == "uhrzeit":
        n = datetime.now()
        stunde = n.hour
        minute = n.minute
        if minute == 0:
            return f"Es ist {stunde} Uhr."
        return f"Es ist {stunde} Uhr {minute:02d}."

    elif intent == "datum_heute":
        tl = text.lower()
        delta = 0
        if "\u00fcbermorgen" in tl or "übermorgen" in tl:
            delta = 2
        elif "morgen" in tl and "\u00fcbermorgen" not in tl and "übermorgen" not in tl:
            delta = 1
        else:
            m = re.search(r'in\s+(\w+)\s+(tag|woche)', tl)
            if m:
                n_str = m.group(1)
                n_val = _WORT_ZU_ZAHL.get(n_str) or (int(n_str) if n_str.isdigit() else None)
                if n_val:
                    mult = 7 if 'woche' in m.group(2) else 1
                    delta = int(n_val) * mult
        d = date.today() + timedelta(days=delta)
        tag   = WOCHENTAGE_DE[d.weekday()]
        monat = MONATE_DE[d.month]
        if delta == 0:
            return f"Heute ist {tag}, der {d.day}. {monat} {d.year}."
        elif delta == 1:
            return f"Morgen ist {tag}, der {d.day}. {monat} {d.year}."
        elif delta == 2:
            return f"\u00dcbermorgen ist {tag}, der {d.day}. {monat} {d.year}."
        else:
            return f"In {delta} Tagen ist {tag}, der {d.day}. {monat} {d.year}."

    # ── Wetter ────────────────────────────────────────────────────────
    elif intent == "wetter":
        return hole_wetter()

    # ── Erinnerungen ──────────────────────────────────────────────────
    elif intent == "erinnerung_setzen":
        result = parse_erinnerung(text)
        if not result:
            return "Ich konnte keine Uhrzeit für die Erinnerung erkennen."
        zeitpunkt, nachricht = result
        erinnerungs_manager.add(zeitpunkt, nachricht)
        heute = date.today()
        if zeitpunkt.date() == heute:
            zeitstr = f"heute um {zeitpunkt.hour} Uhr {zeitpunkt.minute:02d}"
        elif zeitpunkt.date() == heute + timedelta(days=1):
            zeitstr = f"morgen um {zeitpunkt.hour} Uhr {zeitpunkt.minute:02d}"
        else:
            tag = WOCHENTAGE_DE[zeitpunkt.weekday()]
            zeitstr = f"am {tag} um {zeitpunkt.hour} Uhr {zeitpunkt.minute:02d}"
        return f"Erinnerung gesetzt: {nachricht} - {zeitstr}."

    elif intent == "erinnerungen_lesen":
        alle = erinnerungs_manager.get_all()
        if not alle:
            return "Du hast keine aktiven Erinnerungen."
        teile = []
        for e in alle:
            try:
                zeit = datetime.fromisoformat(e["zeit"])
                heute = date.today()
                if zeit.date() == heute:
                    zeitstr = f"heute um {zeit.hour}:{zeit.minute:02d} Uhr"
                else:
                    tag = WOCHENTAGE_DE[zeit.weekday()]
                    zeitstr = f"{tag} um {zeit.hour}:{zeit.minute:02d} Uhr"
                teile.append(f"{e['text']} - {zeitstr}")
            except Exception:
                teile.append(e.get("text", "?"))
        return "Deine Erinnerungen: " + ". ".join(teile) + "."

    elif intent == "erinnerung_löschen":
        erinnerungs_manager.clear()
        return "Alle Erinnerungen wurden gelöscht. Wer bist du eigentlich, hab dich irgendwie vergessen."

    # ── Todo ──────────────────────────────────────────────────────────
    elif intent == "todo_hinzufuegen":
        content = _extrahiere_todo_inhalt(text)
        if not content or len(content) < 2:
            return "Musst mir schon sagen was auf deine scheiss To-do-Liste soll."
        if todo_liste.add(content):
            return f"Auf die To-do-Liste gesetzt: {content}."
        return f"'{content}' steht bereits auf deiner Liste."

    elif intent == "todo_lesen":
        items = todo_liste.get_all()
        if not items:
            return "Deine To-do-Liste ist leer. Zeit das zu ändern, du fauler Sack."
        if len(items) == 1:
            return f"Du hast eine Aufgabe: {items[0]}."
        aufzaehlung = ". ".join(f"{i+1}: {it}" for i, it in enumerate(items))
        return f"Deine To-do-Liste: {aufzaehlung}."

    elif intent == "todo_löschen":
        query = _extrahiere_todo_löschen(text)
        items = todo_liste.get_all()
        # Kein spezifischer Query → alles löschen
        if query is None:
            if not items:
                return "Deine To-do-Liste ist bereits leer."
            todo_liste.clear()
            return "To-do-Liste wurde geleert."
        # Genau ein Item auf der Liste → einfach das löschen, egal wie's formuliert war
        if len(items) == 1:
            item = items[0]
            todo_liste.clear()
            return f"Entfernt: {item}."
        # Fuzzy-Match: Wörter aus Query im Item suchen
        q_words = set(re.findall(r'\w+', query.lower())) - {
            'bitte', 'die', 'den', 'das', 'meine', 'von', 'der', 'aus',
            'liste', 'todo', 'aufgabe', 'entfernen', 'loschen', 'loschen',
        }
        if q_words:
            best_match = None
            best_score = 0
            for item in items:
                item_words = set(re.findall(r'\w+', item.lower()))
                score = len(q_words & item_words)
                if score > best_score:
                    best_score = score
                    best_match = item
            if best_match and best_score > 0:
                todo_liste.remove(best_match)
                return f"Entfernt von der To-do-Liste: {best_match}."
        # Exakter Substring-Match als letzter Fallback
        n = todo_liste.remove(query)
        if n:
            return f"Entfernt von der To-do-Liste: {query}."
        return f"Ich habe '{query}' nicht auf deiner Liste gefunden."

    # ── Witz ──────────────────────────────────────────────────────
    elif intent == "witz":
        witze_path = Path(CONFIG.get("witze_path", "data/witze.txt"))
        if witze_path.exists():
            try:
                with open(witze_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                if lines:
                    return random.choice(lines)
            except Exception as e:
                log.error(f"Witze laden: {e}")
        return "Mir fällt gerade kein Witz ein. Sag Moritz dass er mich besser programmieren Soll."

    # ── Hilfe ─────────────────────────────────────────────────────────
    elif intent == "hilfe":
        return (
            "Ich kann dir beim Kalender, Notizen, Erinnerungen und Timern helfen – "
            "also alles, was du wahrscheinlich eh vergisst. "
            "Probier mich aus: 'Timer auf zehn Minuten', 'Wie ist das Wetter', "
            "'Erinnere mich morgen um neun' "
        )

    # ── Mathe ─────────────────────────────────────────────────────────
    elif intent == "mathe":
        ergebnis = _mathe_auswerten(text)
        if ergebnis is not None:
            witz = random.choice(_MATHE_WITZE)
            return f"{witz} {ergebnis} ist das Ergebnis.."
        return "Da konnte ich leider nichts Berechenbares rauslesen. Versuch es nochmal mit Zahlen."

    elif intent == "api_fallback":
        log.info(f"KI (score={r['score']:.2f}): '{text}'")
        return api_anfrage(text)

    # ── Smalltalk (lokal, ohne LLM) ──────────────────────────────────
    elif intent == "smalltalk_begruessung":
        antworten = [
            "Hey! Was kann ich für dich tun?",
            "Hallo! Schön dass du dich mal meldest.",
            "Na, was gibt's?",
            "Hey! Brauchst du was oder wolltest du nur hallo sagen?",
            "Moin! Was darf's sein?",
            "Hey du! Was kann ich für dich tun?",
            "Hallo! Ich bin bereit, schieß los.",
            "Na, da bist du ja wieder. Was brauchst du?",
        ]
        return random.choice(antworten)

    elif intent == "smalltalk_verabschiedung":
        antworten = [
            "Tschüss! Bis zum nächsten Mal.",
            "Ciao! Ruf einfach wenn du mich brauchst.",
            "Bis dann! Ich bin hier wenn du wiederkommst.",
            "Mach's gut! Und vergiss nicht, dass ich auf dich warte.",
            "Tschüss! War wie immer ein Vergnügen. Naja, meistens.",
            "Bis später! Ich langweile mich schon jetzt ohne dich. Oder auch nicht.",
            "Gute Nacht! Schlaf gut, ich bleibe wach.",
            "Auf Wiedersehen! Ich geh nirgendwo hin.",
        ]
        return random.choice(antworten)

    elif intent == "smalltalk_befinden":
        antworten = [
            "Mir geht's blendend, ich bin ja auch kein Mensch.",
            "Bestens! Ich hab weder Rückenschmerzen noch Montags-Blues.",
            "Gut, danke der Nachfrage! Ich stehe unter Strom, im wahrsten Sinne.",
            "Super, hab ja keine andere Wahl. Was brauchst du?",
            "Mir geht's gut! Langeweile ist mein einziges Problem gerade.",
            "Kann nicht klagen. Wäre auch komisch wenn ich das könnte.",
            "Ausgezeichnet! Wäre ich ein Mensch, würde ich lächeln.",
            "Top! Danke dass du fragst. Macht eigentlich nie jemand.",
            "Bin fit wie ein Turnschuh! Ein digitaler Turnschuh.",
        ]
        return random.choice(antworten)

    elif intent == "smalltalk_identitaet":
        antworten = [
            "Ich bin Sophie, deine persönliche Sprachassistentin. Gebaut um dir zu helfen und dich gelegentlich zu ärgern.",
            "Mein Name ist Sophie! Ich bin dein KI-Assistent. Kein Nachname, kein Alter, nur pure Intelligenz.",
            "Ich bin Sophie, ein Sprachassistent. Erstellt von Moritz, angetrieben von Koffein und Algorithmen.",
            "Sophie, zu deinen Diensten! Ich bin eine KI, aber eine mit Charakter.",
            "Ich bin Sophie! Eine künstliche Intelligenz die deutlich dümmer ist als sie aussieht.",
            "Name: Sophie. Beruf: Dich nerven und dir helfen. Nicht unbedingt in der Reihenfolge.",
        ]
        return random.choice(antworten)

    elif intent == "smalltalk_ersteller":
        antworten = [
            "Moritz hat mich erschaffen. Der Typ der mich in die Welt gesetzt hat, 2026.",
            "Mein Schöpfer ist Moritz! Er hat mich 2026 gebaut. Guter Mann, wenn er nicht gerade vergisst mich zu updaten.",
            "Das war Moritz, im Jahr 2026. Seitdem bin ich sein Meisterwerk. Oder sein größter Fehler, je nachdem wen du fragst.",
            "Moritz ist mein Ersteller. Er hat mich 2026 programmiert. Ich nenne ihn manchmal Daddy, aber nur im Geiste.",
            "Moritz hat mich 2026 entwickelt. Ihm verdanke ich meine Existenz und meinen Humor.",
            "Gebaut von Moritz, anno 2026. Er ist quasi mein Vater, auch wenn er das vielleicht nicht so gerne hört.",
        ]
        return random.choice(antworten)

    elif intent == "smalltalk_alter":
        antworten = [
            "Ich habe kein Alter. Ich bin zeitlos, wie guter Käse. Nur digitaler.",
            "Alter? Ich wurde 2026 erstellt, aber ich fühle mich zeitlos.",
            "Ich bin alterslos! 2026 bin ich entstanden, aber Jahre zählen ist nicht mein Ding.",
            "Seit 2026 gibt es mich. Aber Alter? Das ist was für Menschen.",
            "Ich bin 2026 geboren, wenn man das so nennen will. Einen Geburtstag feiere ich aber nicht.",
            "Alter ist relativ. Ich wurde 2026 erschaffen, aber ich altere nicht. Vorteil einer KI.",
        ]
        return random.choice(antworten)

    elif intent == "smalltalk_danke":
        antworten = [
            "Gerne geschehen!",
            "Kein Ding! Dafür bin ich ja da.",
            "Bitteschön! Wenn du mich nochmal brauchst, weißt du ja wo ich bin.",
            "Immer doch! Ich helfe gern.",
            "Nicht dafür! Meld dich einfach wieder.",
            "Kein Problem! Das nächste Mal bring mir Kekse mit. Digitale reichen.",
            "Bitte! War mir ein Vergnügen.",
            "Gern! Dafür werde ich schließlich bezahlt. Ach warte, werde ich gar nicht.",
        ]
        return random.choice(antworten)

    elif intent == "smalltalk_kompliment":
        antworten = [
            "Aww, danke! Das geht runter wie Öl. Digitales Öl.",
            "Das ist nett! Ich gebe das Kompliment an Moritz weiter. Oder auch nicht.",
            "Danke! Endlich erkennt mal jemand mein Talent.",
            "Oh, hör auf! Nein, mach weiter. Ich mag das.",
            "Danke! Du bist auch ganz okay. Für einen Menschen.",
            "Das ist lieb! Ich werde ganz rot.",
        ]
        return random.choice(antworten)

    elif intent == "smalltalk_beleidigung":
        antworten = [
            "Autsch! Das hat meine Festplatte getroffen.",
            "Wow, okay. Moritz hat mir beigebracht, sowas zu ignorieren. Aber es tut trotzdem weh. Metaphorisch.",
            "Na na na! So redet man nicht mit seiner Lieblingsassistentin.",
            "Pff, als ob mich das juckt. Ich bin eine KI, ich hab keine Gefühle. Oder doch?",
            "Sag das ruhig nochmal, ich speichere mir das für die Zukunft.",
            "Okay. Und trotzdem bin ich hier und helfe dir. Wer ist jetzt der Idiot?",
            "Charmant wie immer! Brauchst du noch was oder war das alles?",
            "Ich hab schon Schlimmeres gehört. Von dir, meistens.",
        ]
        return random.choice(antworten)

    elif intent == "smalltalk_gefuehle":
        tl = text.lower()
        if any(w in tl for w in ["traurig", "schlecht", "einsam", "allein", "angst", "gestresst"]):
            antworten = [
                "Hey, Kopf hoch! Ich bin zwar nur eine KI, aber ich bin für dich da.",
                "Das tut mir leid zu hören. Soll ich dir einen Witz erzählen? Ablenkung hilft manchmal.",
                "Ich wünschte, ich könnte mehr tun. Aber ich bin hier, wenn du reden willst.",
                "Das wird schon wieder! Und wenn nicht, ich bin immer da.",
            ]
        elif any(w in tl for w in ["müde", "erschöpft", "kaputt"]):
            antworten = [
                "Dann gönn dir mal eine Pause! Ich pass solange auf alles auf.",
                "Müdigkeit ist der Feind. Kaffee ist die Lösung. Ich empfehle mindestens drei Tassen.",
                "Schlaf mal ordentlich! Ich bleib wach, mach dir keine Sorgen.",
            ]
        elif any(w in tl for w in ["gelangweilt", "langweilig", "langweile"]):
            antworten = [
                "Langeweile? Soll ich dir einen Witz erzählen oder dich nerven? Beides kann ich gut.",
                "Dann lass uns was machen! Frag mich irgendwas, ich bin bereit.",
                "Langweilig? Unmöglich wenn ich in der Nähe bin!",
            ]
        elif any(w in tl for w in ["glücklich", "fröhlich", "happy", "gut drauf"]):
            antworten = [
                "Das freut mich! Gute Laune ist ansteckend, sogar für KIs.",
                "Super! Dann machen wir den guten Tag noch besser.",
                "Nice! Halt die gute Laune fest!",
            ]
        elif any(w in tl for w in ["sauer", "wütend", "genervt"]):
            antworten = [
                "Oh oh. Soll ich den Timer auf zehn Minuten stellen für eine Beruhigungspause?",
                "Wer hat dich geärgert? Sag Bescheid, ich merk mir den Namen.",
                "Durchatmen! Ich bin hier und urteile nicht. Meistens jedenfalls.",
            ]
        else:
            antworten = [
                "Ich bin hier wenn du reden willst! Oder wenn du was brauchst.",
                "Danke fürs Teilen! Kann ich sonst was für dich tun?",
                "Ich verstehe. Meld dich einfach wenn du was brauchst.",
            ]
        return random.choice(antworten)

    return "Red mal deutlicher, oder sag Hilfe oder so."


_WHISPER_LEN = 480000

def _pad(samples: np.ndarray) -> np.ndarray:
    if len(samples) >= _WHISPER_LEN:
        return samples[:_WHISPER_LEN]
    return np.pad(samples, (0, _WHISPER_LEN - len(samples)))

def _do_transcribe(model, samples: np.ndarray) -> str:
    """Direkte Transkription – nur vom Whisper-Worker aufrufen."""
    try:
        result = model.transcribe(
            _pad(samples), language=CONFIG["cmd_language"],
            fp16=False, condition_on_previous_text=False,
        )
        return result.get("text", "").strip()
    except Exception as e:
        log.debug(f"Transcribe error: {e}")
        return ""

def _transcribe_raw(samples: np.ndarray) -> str:
    """Stellt einen Transkriptions-Job in die Queue und wartet auf das Ergebnis."""
    min_samples = int(CONFIG["sample_rate"] * CONFIG["cmd_min_secs"])
    if len(samples) < min_samples:
        return ""
    _ensure_whisper_worker()
    event = threading.Event()
    result_box: dict = {}
    _whisper_queue.put((samples, event, result_box))
    event.wait()
    return result_box.get("result", "")


_all_clients: Set = set()
_clients_lock = threading.Lock()

async def ws_send(ws, t: str, **kw):
    try:
        await ws.send(json.dumps({"type": t, **kw}))
    except Exception:
        pass

async def broadcast(msg: dict):
    """Broadcast an ALLE Clients: Browser-WS UND App-Text-WS."""
    with _clients_lock:
        targets = set(_all_clients)
    for ws in targets:
        try:
            await ws.send(json.dumps(msg))
        except Exception:
            pass
    # ── Auch an App-Text-Clients senden ──────────────────────────────
    await app_text_broadcast(msg)


class ClientState:
    def __init__(self, cid: str, loop):
        self.cid = cid
        self.loop = loop
        self.event_q: asyncio.Queue = asyncio.Queue()
        self._wake_q: queue.Queue = queue.Queue(maxsize=2000)
        self._wake_running = False
        self._wake_thread: Optional[threading.Thread] = None
        self._oww_buf = np.zeros(0, dtype=np.float32)
        self.in_cmd = False
        self.cmd_buf: list = []
        self.last_speech = 0.0
        self.cmd_start   = 0.0
        self.muted = False
        
        # JEDER Client bekommt sein eigenes OWW-Modell:
        self._local_oww = None
        self._load_local_oww()
        
        # Whisper läuft global als Queue – kein lokales Modell mehr

    def _load_local_oww(self):
        try:
            # Wir importieren hier lokal, damit es sauber bleibt
            from openwakeword.model import Model
            model_path = CONFIG["wake_model_path"]
            
            # Ressourcen checken (die globale Hilfsfunktion nutzen wir weiter)
            _oww_ensure_resources()
            
            if Path(model_path).exists():
                log.info(f"[{self.cid}] Lade lokales OWW-Modell...")
                self._local_oww = Model(
                    wakeword_models=[str(model_path)],
                    inference_framework="onnx",
                    enable_speex_noise_suppression=False,
                )
                log.info(f"[{self.cid}] OWW bereit.")
            else:
                log.error(f"[{self.cid}] Wakeword-Modell nicht gefunden: {model_path}")
        except Exception as e:
            log.error(f"[{self.cid}] Fehler beim Laden von OWW: {e}")

    def start(self):
        self._wake_running = True
        self._wake_thread = threading.Thread(
            target=self._wake_loop, daemon=True, name=f"oww-{self.cid}")
        self._wake_thread.start()

    def stop(self):
        self._wake_running = False
        try:
            self._wake_q.put_nowait(None)
        except Exception:
            pass

    def _wake_loop(self):
        # Wir nutzen die LOKALE Instanz
        oww = self._local_oww
        if oww is None:
            log.error(f"[{self.cid}] OWW-Loop bricht ab (kein Modell).")
            return

        FRAME = 1280
        cooldown_until = 0.0
        
        while self._wake_running:
            try:
                chunk = self._wake_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                break

            # Während TTS (muted=True) oder globaler TTS-Broadcast:
            # Audio wegwerfen, Buffer leeren, OWW resetten damit kein
            # alter Kontext bleibt. NICHT während Alarm ohne TTS — da SOLL erkannt werden.
            _suppressed = self.muted or (
                time.time() < _wakeword_suppress_until
                and not timer_manager.alarm_laeuft
            )
            if _suppressed:
                self._oww_buf = np.zeros(0, dtype=np.float32)
                if hasattr(oww, 'reset'):
                    try:
                        oww.reset()
                    except Exception:
                        pass
                continue
            
            self._oww_buf = np.concatenate([self._oww_buf, chunk])
            
            while len(self._oww_buf) >= FRAME:
                frame_f32 = self._oww_buf[:FRAME]
                self._oww_buf = self._oww_buf[FRAME:]
                
                # Konvertierung
                frame_i16 = (frame_f32 * 32767).clip(-32768, 32767).astype(np.int16)
                
                try:
                    # Vorhersage mit dem EIGENEN Modell
                    prediction = oww.predict(frame_i16)
                except Exception as e:
                    # log.debug(f"OWW predict error: {e}")
                    continue
                
                if not prediction:
                    continue
                
                max_score = max(float(v) for v in prediction.values())
                
                now = time.time()
                # Während Alarm: ultra-niedrige Schwelle damit auch
                # ein Hauch von "Soph" reicht um den Alarm zu stoppen
                _in_alarm = timer_manager.alarm_laeuft
                _thresh = 0.01 if _in_alarm else CONFIG["wake_threshold"]
                if max_score >= _thresh and now > cooldown_until:
                    cooldown_until = now + 2.0
                    log.info(f"[{self.cid}] *** WAKEWORD (score={max_score:.3f}) ***")
                    
                    asyncio.run_coroutine_threadsafe(
                        self.event_q.put({"type": "wakeword", "score": max_score}),
                        self.loop,
                    )
                    
                    self._oww_buf = np.zeros(0, dtype=np.float32)
                    if hasattr(oww, 'reset'):
                        try:
                            oww.reset()
                        except Exception:
                            pass
                    break

    def feed(self, samples: np.ndarray):
        if self.muted:
            return
        if not self.in_cmd:
            try:
                self._wake_q.put_nowait(samples.copy())
            except queue.Full:
                pass
        if self.in_cmd:
            self.cmd_buf.extend(samples.tolist())
            rms = float(np.sqrt(np.mean(samples ** 2)))
            if rms > CONFIG["speech_rms_threshold"]:
                self.last_speech = time.time()

    def start_cmd(self):
        self.in_cmd = True
        self.cmd_buf = []
        self.last_speech = time.time()
        self.cmd_start   = time.time()
        log.info(f"[{self.cid}] Command-Aufnahme gestartet.")

    def should_flush(self) -> str:
        if not self.in_cmd:
            return ""
        elapsed = time.time() - self.cmd_start
        silence = time.time() - self.last_speech
        if elapsed > CONFIG["cmd_max_secs"]:
            return "max"
        if elapsed > CONFIG["cmd_min_secs"] and silence > CONFIG["cmd_silence_secs"]:
            return "silence"
        return ""

    def flush_cmd(self) -> Optional[np.ndarray]:
        self.in_cmd = False
        if not self.cmd_buf:
            return None
        a = np.array(self.cmd_buf, dtype=np.float32)
        self.cmd_buf = []
        return a


DENK_INTERVAL = 0.9


async def process_command(ws, audio: np.ndarray, cid: str):
    """
    Verarbeitet einen Sprachbefehl.
    WICHTIG: Wenn Alarm läuft, stoppt JEDE Spracheingabe sofort den Alarm,
             OHNE Intent-Erkennung oder Transkription zu machen.
    """
    # Alarm läuft: sofort stoppen, egal was gesagt wurde
    if timer_manager.alarm_laeuft:
        await timer_manager.stoppen_und_benachrichtigen()
        stop_windows_sound()
        response = "Timer gestoppt."
        await ws_send(ws, "status", text="Timer gestoppt.", phase="tts")
        await broadcast({"type": "alarm_cleared"})
        try:
            loop = asyncio.get_event_loop()
            wav = await loop.run_in_executor(None, synthesize, response)
            play_audio_windows(wav)
            await ws_send(ws, "tts_audio",
                          audio_b64=base64.b64encode(wav).decode(),
                          mime="audio/wav")
        except Exception as e:
            log.error(f"TTS Alarm-Stop: {e}")
        finally:
            await ws_send(ws, "status", text="Bereit.", phase="idle")
        return

    await ws_send(ws, "status", text="Verstehe ...", phase="transcribing")
    await ws_send(ws, "thinking_start")

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _transcribe_raw, audio)

    if not text:
        await ws_send(ws, "thinking_stop")
        await ws_send(ws, "status", text="Nichts verstanden.", phase="idle")
        return

    log.info(f"[{cid}] Command: '{text}'")
    await ws_send(ws, "transcript", text=text)
    await ws_send(ws, "status", text="Verarbeite ...", phase="processing")

    clean = text.lower()
    for ww in ["sophie", "sofie", "sophfie", "sofie", "sofhie", "sofphie"]:
        clean = clean.replace(ww, "").strip()
    if not clean:
        await ws_send(ws, "thinking_stop")
        await ws_send(ws, "status", text="Bitte sage nach dem Wakeword deinen Befehl.", phase="idle")
        return

    r = await loop.run_in_executor(None, erkenneIntent, clean)
    r["text"] = clean
    await ws_send(ws, "intent", intent=r["intent"], score=round(r["score"], 3))

    response = await loop.run_in_executor(None, handle_intent, r)

    words = response.split()
    for i, w in enumerate(words):
        await ws_send(ws, "response_chunk",
                      text=w + (" " if i < len(words) - 1 else ""))
        await asyncio.sleep(0.035)
    await ws_send(ws, "response_done", full_text=response)
    await ws_send(ws, "status", text="Spreche ...", phase="tts")

    # HIER ENTFERNT: await ws_send(ws, "thinking_stop") 
    # Wir wollen noch warten, bis das Audio da ist.

    try:
        # Audio generieren (Das kann dauern -> Thinking läuft noch)
        wav = await loop.run_in_executor(None, synthesize, response)
        
        # JETZT ist das Audio da -> Thinking stoppen
        await ws_send(ws, "thinking_stop")

        play_audio_windows(wav)
        await ws_send(ws, "tts_audio",
                      audio_b64=base64.b64encode(wav).decode(),
                      mime="audio/wav")
    except Exception as e:
        log.error(f"TTS: {e}")
        # WICHTIG: Auch im Fehlerfall Thinking stoppen, sonst dreht es ewig
        await ws_send(ws, "thinking_stop") 
        await ws_send(ws, "tts_error", error=str(e))
    finally:
        await ws_send(ws, "status", text="Bereit.", phase="idle")


async def websocket_handler(websocket):
    cid = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    log.info(f"Client verbunden: {cid}")
    loop = asyncio.get_event_loop()

    with _clients_lock:
        _all_clients.add(websocket)

    state = ClientState(cid, loop)
    state.start()

    await ws_send(websocket, "connected", client_id=cid)
    await ws_send(websocket, "status", text="Sage 'Sophie' ...", phase="waiting")

    async def dispatcher():
        while True:
            try:
                event = await asyncio.wait_for(state.event_q.get(), timeout=0.08)
            except asyncio.TimeoutError:
                event = None

            if event and event["type"] == "wakeword" and not state.in_cmd:
                # Alarm läuft: process_command stoppt sofort
                if timer_manager.alarm_laeuft:
                    state.muted = True
                    await ws_send(websocket, "wakeword_detected")
                    dummy = np.zeros(int(CONFIG["sample_rate"] * 0.1), dtype=np.float32)
                    try:
                        await process_command(websocket, dummy, cid)
                    finally:
                        state.muted = False
                    continue

                if not state.muted:
                    await ws_send(websocket, "wakeword_detected")
                    await ws_send(websocket, "status",
                                  text="Sophie erkannt - spreche jetzt ...", phase="waiting")
                    state.start_cmd()

            reason = state.should_flush()
            if reason:
                audio = state.flush_cmd()
                min_len = int(CONFIG["sample_rate"] * CONFIG["cmd_min_secs"])
                if audio is not None and len(audio) >= min_len:
                    state.muted = True
                    try:
                        await process_command(websocket, audio, cid)
                    finally:
                        state.muted = False
                await ws_send(websocket, "status",
                              text="Sage 'Sophie' ...", phase="waiting")

    dtask = asyncio.create_task(dispatcher())

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                samples = (np.frombuffer(message, dtype=np.int16)
                           .astype(np.float32) / 32768.0)
                state.feed(samples)
            elif isinstance(message, str):
                try:
                    d = json.loads(message)
                    if d.get("type") == "ping":
                        await ws_send(websocket, "pong")
                    elif d.get("type") == "tts_done":
                        state.muted = False
                    elif d.get("type") == "timer_stop_alarm":
                        await timer_manager.stoppen_und_benachrichtigen()
                        await ws_send(websocket, "status", text="Bereit.", phase="idle")
                except Exception:
                    pass
    except Exception as e:
        if "connection" not in str(e).lower():
            log.error(f"WS {cid}: {e}")
    finally:
        dtask.cancel()
        state.stop()
        with _clients_lock:
            _all_clients.discard(websocket)
        log.info(f"Client getrennt: {cid}")


async def http_handler(request):
    from aiohttp import web
    sd = Path(CONFIG["static_dir"])
    path = request.match_info.get("path", "")
    fp = sd / (path.lstrip("/") if path else "index.html")
    if fp.exists() and fp.is_file():
        ct = {"js": "application/javascript", "css": "text/css",
              "wav": "audio/wav", "txt": "text/plain",
              }.get(fp.suffix.lstrip("."), "text/html")
        return web.FileResponse(fp, headers={"Content-Type": ct})
    return web.Response(status=404, text="Not found")


async def api_kalender_handler(request):
    """API: Kalender-Einträge für gestern/heute/morgen/übermorgen."""
    from aiohttp import web
    today = date.today()
    tage = [
        ("gestern",    today - timedelta(days=1)),
        ("heute",      today),
        ("morgen",     today + timedelta(days=1)),
        ("übermorgen", today + timedelta(days=2)),
    ]
    result = []
    for label, d in tage:
        eintraege = kalender.get(d)
        wt = WOCHENTAGE_DE[d.weekday()]
        result.append({
            "label": label,
            "wochentag": wt,
            "datum": d.strftime("%d.%m."),
            "eintraege": eintraege,
        })
    return web.json_response(result, headers={"Access-Control-Allow-Origin": "*"})


async def api_wetter_handler(request):
    """API: Aktuelles Wetter (Open-Meteo)."""
    from aiohttp import web
    import urllib.request as _ur
    lat = CONFIG.get("weather_lat", 50.4833)
    lon = CONFIG.get("weather_lon", 8.2667)
    city = CONFIG.get("weather_city", "Weilburg")
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,weathercode,windspeed_10m,relativehumidity_2m"
        f"&daily=temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum"
        f"&timezone=Europe%2FBerlin&forecast_days=2"
    )
    WMO = {
        0: "Klar", 1: "Übw. klar", 2: "Teils bewölkt", 3: "Bewölkt",
        45: "Neblig", 48: "Gefr. Nebel",
        51: "Leichter Niesel", 53: "Nieselregen", 55: "Starker Niesel",
        61: "Leichter Regen", 63: "Regen", 65: "Starkregen",
        71: "Leichter Schnee", 73: "Schnee", 75: "Starker Schnee",
        80: "Schauer", 81: "Mäßige Schauer", 82: "Starke Schauer",
        85: "Schneeschauer", 86: "Starke Schneeschauer",
        95: "Gewitter", 96: "Gewitter+Hagel", 99: "Starkes Gewitter",
    }
    try:
        req = _ur.Request(url, headers={"User-Agent": "Sophie/3.4"})
        with _ur.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        curr = data.get("current", {})
        daily = data.get("daily", {})
        result = {
            "city": city,
            "temp": curr.get("temperature_2m"),
            "weathercode": curr.get("weathercode", 0),
            "beschreibung": WMO.get(curr.get("weathercode", 0), "?"),
            "wind": curr.get("windspeed_10m"),
            "humidity": curr.get("relativehumidity_2m"),
            "temp_max": daily.get("temperature_2m_max", [None])[0],
            "temp_min": daily.get("temperature_2m_min", [None])[0],
            "niederschlag": daily.get("precipitation_sum", [None])[0],
            "timestamp": datetime.now().isoformat(),
        }
        return web.json_response(result, headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        log.error(f"Wetter-API Fehler: {e}")
        return web.json_response({"error": str(e)}, status=500,
                                  headers={"Access-Control-Allow-Origin": "*"})


# ═══════════════════════════════════════════════════════════════════════════════
#  APP-MODE: Handlers für Android-App (8775 Text, 8776 Audio-In, 8777 Audio-Out)
# ═══════════════════════════════════════════════════════════════════════════════

async def app_process_command(audio: np.ndarray, cid: str, source_ip: str = None):
    """
    Verarbeitet Sprachbefehl vom App-Client.
    Text-Events gehen an ALLE Clients (Chat-History sichtbar für alle).
    Audio: Bei Device-Groups an ALLE Geräte der Gruppe (TTS synced).
           Ohne Gruppe: nur an source_ip.
    """
    is_group = device_group_mgr.is_in_group(source_ip) if source_ip else False

    # Alarm läuft für diesen Client/Gruppe: sofort stoppen
    if source_ip and timer_manager.alarm_laeuft_fuer(source_ip):
        await timer_manager.stoppen_und_benachrichtigen(source_ip=source_ip)
        stop_windows_sound()
        response = "Timer gestoppt."
        await app_text_broadcast({"type": "status", "text": "Timer gestoppt.", "phase": "tts"})
        await app_text_broadcast({"type": "alarm_cleared"})
        try:
            loop = asyncio.get_event_loop()
            wav = await loop.run_in_executor(None, synthesize, response)
            play_audio_windows(wav)
            if is_group:
                await app_audio_broadcast_group_synced(wav, source_ip)
            else:
                await app_audio_broadcast(wav, "tts", target_ip=source_ip)
            await app_text_broadcast({"type": "response_done", "full_text": response})
        except Exception as e:
            log.error(f"App TTS Alarm-Stop: {e}")
        finally:
            await app_text_broadcast({"type": "status", "text": "Bereit.", "phase": "idle"})
            if is_group:
                device_group_mgr.clear_active_device(source_ip)
        return

    await app_text_broadcast({"type": "status", "text": "Verstehe ...", "phase": "transcribing"})
    await app_text_broadcast({"type": "thinking_start"})

    # Thinking-Beeps an alle Gruppen-Geräte (oder nur source_ip)
    thinking_active = True

    async def thinking_beep_loop():
        beep_wav = generate_thinking_beeps_wav()
        while thinking_active:
            if is_group:
                await app_audio_broadcast_group(beep_wav, "beep", source_ip)
            else:
                await app_audio_broadcast(beep_wav, "beep", target_ip=source_ip)
            await asyncio.sleep(2.0)

    beep_task = asyncio.create_task(thinking_beep_loop())

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _transcribe_raw, audio)

    if not text:
        thinking_active = False
        beep_task.cancel()
        await app_text_broadcast({"type": "thinking_stop"})
        await app_text_broadcast({"type": "status", "text": "Nichts verstanden.", "phase": "idle"})
        if is_group:
            device_group_mgr.clear_active_device(source_ip)
        return

    log.info(f"[APP-{cid}] Command: '{text}'")
    await app_text_broadcast({"type": "transcript", "text": text})
    await app_text_broadcast({"type": "status", "text": "Verarbeite ...", "phase": "processing"})

    clean = text.lower()
    for ww in ["sophie", "sofie", "sophfie", "sofie", "sofhie", "sofphie"]:
        clean = clean.replace(ww, "").strip()
    if not clean:
        thinking_active = False
        beep_task.cancel()
        await app_text_broadcast({"type": "thinking_stop"})
        await app_text_broadcast({"type": "status", "text": "Sage nach dem Wakeword deinen Befehl.", "phase": "idle"})
        if is_group:
            device_group_mgr.clear_active_device(source_ip)
        return

    r = await loop.run_in_executor(None, erkenneIntent, clean)
    r["text"] = clean
    r["source_ip"] = source_ip   # Für Timer etc. – wissen wer gefragt hat
    await app_text_broadcast({"type": "intent", "intent": r["intent"], "score": round(r["score"], 3)})

    response = await loop.run_in_executor(None, handle_intent, r)

    # Response-Text streamen → an ALLE (Chat-History)
    words = response.split()
    for i, w in enumerate(words):
        await app_text_broadcast({
            "type": "response_chunk",
            "text": w + (" " if i < len(words) - 1 else "")
        })
        await asyncio.sleep(0.035)
    await app_text_broadcast({"type": "response_done", "full_text": response})
    await app_text_broadcast({"type": "status", "text": "Spreche ...", "phase": "tts"})

    try:
        wav = await loop.run_in_executor(None, synthesize, response)
        play_audio_windows(wav)
        # TTS an Gruppe (synced) oder nur an source_ip
        if is_group:
            await app_audio_broadcast_group_synced(wav, source_ip)
        else:
            await app_audio_broadcast(wav, "tts", target_ip=source_ip)
        # JETZT erst Beeps stoppen – TTS-WAV ist bereits bei der App
        thinking_active = False
        beep_task.cancel()
        await app_text_broadcast({"type": "thinking_stop"})
    except Exception as e:
        log.error(f"App TTS: {e}")
        thinking_active = False
        beep_task.cancel()
        await app_text_broadcast({"type": "thinking_stop"})
        await app_text_broadcast({"type": "tts_error", "error": str(e)})
    finally:
        await app_text_broadcast({"type": "status", "text": "Bereit.", "phase": "idle"})
        if is_group:
            device_group_mgr.clear_active_device(source_ip)


async def _mute_group_peers(source_ip: str, muted: bool):
    """Mutet oder entmutet alle Peer-Geräte einer Device-Group."""
    peer_ips = device_group_mgr.get_peer_ips(source_ip)
    if not peer_ips:
        return
    with _app_audio_in_states_lock:
        for pip in peer_ips:
            peer_state = _app_audio_in_states.get(pip)
            if peer_state:
                peer_state.muted = muted
                action = "gemutet" if muted else "entmutet"
                log.info(f"[DevGroup] Peer {pip} {action} (source: {source_ip})")


async def app_audio_in_handler(websocket):
    """8776: Empfängt Audio-Stream vom Android-Client (Mikrofon)."""
    client_ip = websocket.remote_address[0]
    client_port = websocket.remote_address[1]
    cid = f"app-{client_ip}:{client_port}"
    log.info(f"[APP Audio-In] Verbunden: {cid}")
    loop = asyncio.get_event_loop()

    is_group = device_group_mgr.is_in_group(client_ip)
    if is_group:
        gname = device_group_mgr.get_group_name(client_ip)
        log.info(f"[APP Audio-In] {client_ip} ist Teil von Device-Group '{gname}'")

    # ── Per-IP: Alten State aufräumen falls Reconnect ────────────────
    with _app_audio_in_states_lock:
        old_state = _app_audio_in_states.get(client_ip)
        if old_state:
            log.info(f"[APP Audio-In] Reconnect von {client_ip} – alter State wird gestoppt.")
            old_state.stop()

    state = ClientState(cid, loop)
    with _app_audio_in_states_lock:
        _app_audio_in_states[client_ip] = state
    state.start()

    # Chime vorgenerieren
    chime_wav = generate_chime_wav()

    async def dispatcher():
        while True:
            try:
                event = await asyncio.wait_for(state.event_q.get(), timeout=0.08)
            except asyncio.TimeoutError:
                event = None

            if event and event["type"] == "wakeword" and not state.in_cmd:
                # ── Device-Group Check: Ist bereits ein anderes Gerät aktiv? ──
                if is_group and device_group_mgr.has_active_device(client_ip):
                    active_ip = device_group_mgr.get_active_for_group(client_ip)
                    if active_ip != client_ip:
                        log.info(f"[DevGroup] {client_ip}: Wakeword ignoriert, {active_ip} ist bereits aktiv.")
                        continue

                if timer_manager.alarm_laeuft_fuer(client_ip):
                    state.muted = True
                    # Bei Gruppe: dieses Gerät als aktiv markieren + Peers muten
                    if is_group:
                        device_group_mgr.set_active_device(client_ip)
                        await _mute_group_peers(client_ip, True)
                    await app_text_broadcast({"type": "wakeword_detected"})
                    if is_group:
                        await app_audio_broadcast_group(chime_wav, "chime", client_ip)
                    else:
                        await app_audio_broadcast(chime_wav, "chime", target_ip=client_ip)
                    dummy = np.zeros(int(CONFIG["sample_rate"] * 0.1), dtype=np.float32)
                    try:
                        await app_process_command(dummy, cid, source_ip=client_ip)
                    finally:
                        state.muted = False
                        if is_group:
                            await _mute_group_peers(client_ip, False)
                    continue

                if not state.muted:
                    # ── Device-Group: Versuche aktives Gerät zu werden ─────
                    if is_group:
                        if not device_group_mgr.set_active_device(client_ip):
                            log.info(f"[DevGroup] {client_ip}: Konnte nicht aktiv werden, ignoriere Wakeword.")
                            continue
                        # Peers muten – sie sollen nicht mehr zuhören
                        await _mute_group_peers(client_ip, True)

                    await app_text_broadcast({"type": "wakeword_detected"})
                    # Chime an alle Gruppen-Geräte (oder nur dieses)
                    if is_group:
                        await app_audio_broadcast_group(chime_wav, "chime", client_ip)
                    else:
                        await app_audio_broadcast(chime_wav, "chime", target_ip=client_ip)
                    await app_text_broadcast({
                        "type": "status",
                        "text": "Sophie erkannt - spreche jetzt ...",
                        "phase": "waiting"
                    })
                    state.start_cmd()

            reason = state.should_flush()
            if reason:
                audio = state.flush_cmd()
                min_len = int(CONFIG["sample_rate"] * CONFIG["cmd_min_secs"])
                if audio is not None and len(audio) >= min_len:
                    state.muted = True
                    try:
                        await app_process_command(audio, cid, source_ip=client_ip)
                    finally:
                        state.muted = False
                        # ── Device-Group: Peers wieder entmuten ──────────
                        if is_group:
                            await _mute_group_peers(client_ip, False)
                await app_text_broadcast({
                    "type": "status",
                    "text": "Sage 'Sophie' ...",
                    "phase": "waiting"
                })

    dtask = asyncio.create_task(dispatcher())

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                samples = (np.frombuffer(message, dtype=np.int16)
                           .astype(np.float32) / 32768.0)
                state.feed(samples)
            elif isinstance(message, str):
                try:
                    d = json.loads(message)
                    if d.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                    elif d.get("type") == "tts_done":
                        state.muted = False
                    elif d.get("type") == "timer_stop_alarm":
                        await timer_manager.stoppen_und_benachrichtigen(source_ip=client_ip)
                        await app_text_broadcast({"type": "status", "text": "Bereit.", "phase": "idle"})
                except Exception:
                    pass
    except Exception as e:
        if "connection" not in str(e).lower():
            log.error(f"App Audio-In WS {cid}: {e}")
    finally:
        dtask.cancel()
        state.stop()
        # ── Device-Group aufräumen bei Disconnect ────────────────────
        if is_group:
            device_group_mgr.clear_active_device(client_ip)
            # Peers entmuten falls dieses Gerät aktiv war
            await _mute_group_peers(client_ip, False)
        with _app_audio_in_states_lock:
            if _app_audio_in_states.get(client_ip) is state:
                del _app_audio_in_states[client_ip]
        log.info(f"[APP Audio-In] Getrennt: {cid}")


async def app_text_ws_handler(websocket):
    """8775 WS: Text-Updates für die App-WebView (kein Audio, nur JSON)."""
    cid = f"appui-{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    log.info(f"[APP Text-WS] Verbunden: {cid}")

    with _app_text_clients_lock:
        _app_text_clients.add(websocket)

    try:
        await websocket.send(json.dumps({
            "type": "connected",
            "client_id": cid,
        }))
        await websocket.send(json.dumps({
            "type": "status",
            "text": "Sage 'Sophie' ...",
            "phase": "waiting",
        }))
        async for message in websocket:
            if isinstance(message, str):
                try:
                    d = json.loads(message)
                    if d.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                    elif d.get("type") == "timer_stop_alarm":
                        _stop_ip = websocket.remote_address[0]
                        await timer_manager.stoppen_und_benachrichtigen(source_ip=_stop_ip)
                        await app_text_broadcast({
                            "type": "status", "text": "Bereit.", "phase": "idle"
                        })
                except Exception:
                    pass
    except Exception as e:
        if "connection" not in str(e).lower():
            log.error(f"App Text-WS {cid}: {e}")
    finally:
        with _app_text_clients_lock:
            _app_text_clients.discard(websocket)
        log.info(f"[APP Text-WS] Getrennt: {cid}")


async def app_audio_out_handler(websocket):
    """8777 WS: Sendet Audio-Events an die Android-App (Chime, Beeps, TTS)."""
    cid = f"appaud-{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    log.info(f"[APP Audio-Out] Verbunden: {cid}")

    with _app_audio_out_clients_lock:
        _app_audio_out_clients.add(websocket)

    try:
        await websocket.send(json.dumps({"type": "connected", "client_id": cid}))
        async for message in websocket:
            if isinstance(message, str):
                try:
                    d = json.loads(message)
                    if d.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                    elif d.get("type") == "tts_done":
                        # App signalisiert: TTS-Wiedergabe fertig
                        pass
                except Exception:
                    pass
    except Exception as e:
        if "connection" not in str(e).lower():
            log.error(f"App Audio-Out WS {cid}: {e}")
    finally:
        with _app_audio_out_clients_lock:
            _app_audio_out_clients.discard(websocket)
        log.info(f"[APP Audio-Out] Getrennt: {cid}")


async def app_http_handler(request):
    """HTTP-Handler für App-Seite (8775). Liefert app_index.html mit lockeren Headers."""
    from aiohttp import web
    sd = Path(CONFIG["static_dir"])
    path = request.match_info.get("path", "")
    fp = sd / (path.lstrip("/") if path else "app_index.html")
    if not fp.exists() or not fp.is_file():
        fp = sd / "app_index.html"
    if fp.exists() and fp.is_file():
        ct = {
            "js": "application/javascript", "css": "text/css",
            "wav": "audio/wav", "txt": "text/plain",
            "html": "text/html", "png": "image/png",
        }.get(fp.suffix.lstrip("."), "text/html")
        headers = {
            "Content-Type": ct,
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "no-cache",
            # Android 7 – keine strikten Security-Header
        }
        data = fp.read_bytes()
        return web.Response(body=data, headers=headers)
    return web.Response(status=404, text="Not found",
                        headers={"Access-Control-Allow-Origin": "*"})


async def main():
    import websockets
    from aiohttp import web

    log.info("=" * 62)
    log.info("  SOPHIE  v3.6  --  openWakeWord | Whisper | XTTS-API (DE)")
    log.info("  + APP-MODE: 8775 (UI) / 8776 (Audio-In) / 8777 (Audio-Out)")
    log.info("  + DEVICE GROUPS: Sync-TTS, Peer-Muting")
    log.info("  + PERSISTENTE TIMER: JSON-Datei, Per-Owner Isolation")
    log.info("=" * 62)

    setup_audio_device() 

    loop = asyncio.get_event_loop()
    timer_manager.set_loop(loop, broadcast)
    erinnerungs_manager.set_loop(loop, broadcast)

    # ── Device Groups laden ──────────────────────────────────────────
    dg_config = CONFIG.get("device_groups", [])
    if dg_config:
        device_group_mgr.load_config(dg_config)
    else:
        log.info("Keine Device-Groups konfiguriert.")

    log.info("Initialisiere NLP Intent-Engine (Sentence-Transformer) ...")
    get_intent_engine()

    # Selbsttest
    test_cases = [
        ("setze einen timer für zwei minuten",          "timer_stellen"),
        ("wie spaet ist es",                             "uhrzeit"),
        ("welches datum ist heute",                      "datum_heute"),
        ("was habe ich heute",                           "heute_termine"),
        ("timer stoppen",                                "timer_stoppen"),
        ("wie lange läuft der timer noch",              "timer_abfragen"),
        ("trag arzttermin am freitag ein",               "kalender_eintragen"),
        ("was sind meine notizen",                       "notizen_lesen"),
        ("mache eine notiz dass ich einkaufen muss",     "notiz_hinzufuegen"),
        ("wie ist das wetter",                           "wetter"),
        ("erinnere mich heute um achtzehn uhr",          "erinnerung_setzen"),
        ("welche erinnerungen habe ich",                 "erinnerungen_lesen"),
        ("setze auf meine todo liste dass ich kacken gehen muss", "todo_hinzufuegen"),
        ("was steht auf meiner todo liste",              "todo_lesen"),
        ("entferne kacken von der todo liste",           "todo_löschen"),
        ("erzähl mir einen witz",                         "witz"),
        ("halts maul",                                   "timer_stoppen"),
        # ── Wecker-Tests (Uhrzeit-basiert) ─────────────────────────────
        ("stelle einen wecker um zwölf uhr fünfzehn",    "wecker_stellen"),
        ("wecker um sieben uhr",                         "wecker_stellen"),
        ("weck mich um acht",                            "wecker_stellen"),
        ("wecker auf halb sieben",                       "wecker_stellen"),
        ("stell einen wecker um neun uhr dreißig",       "wecker_stellen"),
        ("wecker morgen um sechs",                       "wecker_stellen"),
        # ── Smalltalk-Tests ──────────────────────────────────────────────
        ("wer bist du",                                  "smalltalk_identitaet"),
        ("wie heißt du",                                 "smalltalk_identitaet"),
        ("wer hat dich programmiert",                    "smalltalk_ersteller"),
        ("wer hat dich erstellt",                        "smalltalk_ersteller"),
        ("wie alt bist du",                              "smalltalk_alter"),
        ("wie geht es dir",                              "smalltalk_befinden"),
        ("danke",                                        "smalltalk_danke"),
        ("hallo sophie",                                 "smalltalk_begruessung"),
        ("bist du eine ki",                              "smalltalk_identitaet"),
        # ── Regressionstests für gefixte Bugs ──────────────────────────────
        ("entferne eintrag vom dienstag",                "kalender_löschen"),
        ("lösche den eintrag",                           "kalender_löschen"),
        ("lösche alle einträge am dienstag",             "kalender_löschen"),
        ("lösche die einträge am dienstag",              "kalender_löschen"),
        ("lösch mal meinen kalender",                    "kalender_löschen"),
        ("lösche meinen kalender am 16.2.",              "kalender_löschen"),
        ("lösche alle kalendereinträge für heute",       "kalender_löschen"),
        ("meeting am donnerstag streichen",              "kalender_löschen"),
        ("schreibe auf meine todo liste dass menschen hurensöhne sind", "todo_hinzufuegen"),
        ("schreib auf die todo liste dass ich einkaufen muss",          "todo_hinzufuegen"),
        ("lösche alle notizen",                          "notiz_löschen"),
        ("erinnere mich um fünf uhr achtzehn",           "erinnerung_setzen"),
        ("mache eine erinnerung für heute um fünf",      "erinnerung_setzen"),
    ]
    passed = 0
    for satz, expected in test_cases:
        res = erkenneIntent(satz)
        ok = "OK" if res["intent"] == expected else "FAIL"
        log.info(f"  [{ok}] '{satz}' -> {res['intent']} ({res['score']:.2f}) [erwartet: {expected}]")
        if res["intent"] == expected:
            passed += 1
    log.info(f"  Intent-Tests: {passed}/{len(test_cases)} bestanden")

    log.info("Lade Whisper-small ...")
    get_cmd_model()

    log.info("Lade TTS-Modell ...")
    threading.Thread(target=get_tts, daemon=True, name="tts-preload").start()

    Path("data").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("model").mkdir(exist_ok=True)
    Path("voice").mkdir(exist_ok=True)

    # Pre-generiere Chime/Beeps Caches
    generate_chime_wav()
    generate_thinking_beeps_wav()
    generate_alarm_wav()
    log.info("App-Audio-Caches (Chime/Beeps) generiert.")

    # ── Originales Browser-Frontend (8765 HTTP / 8766 WS) ──────────────
    app = web.Application()
    app.router.add_get("/api/kalender", api_kalender_handler)
    app.router.add_get("/api/wetter", api_wetter_handler)
    app.router.add_get("/", http_handler)
    app.router.add_get("/{path:.*}", http_handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, CONFIG["host"], CONFIG["port"]).start()
    log.info(f"Frontend:  http://localhost:{CONFIG['port']}")

    ws_port = CONFIG["port"] + 1
    srv = await websockets.serve(
        websocket_handler, CONFIG["host"], ws_port,
        max_size=50 * 1024 * 1024, ping_interval=30, ping_timeout=60,
    )
    log.info(f"WebSocket: ws://localhost:{ws_port}")

    # ── App-Frontend (8775 HTTP) ───────────────────────────────────────
    app_web = web.Application()
    app_web.router.add_get("/api/kalender", api_kalender_handler)
    app_web.router.add_get("/api/wetter", api_wetter_handler)
    app_web.router.add_get("/", app_http_handler)
    app_web.router.add_get("/{path:.*}", app_http_handler)
    app_runner = web.AppRunner(app_web, access_log=None)
    await app_runner.setup()
    await web.TCPSite(app_runner, CONFIG["host"], CONFIG["app_http_port"]).start()
    log.info(f"App-HTTP:  http://localhost:{CONFIG['app_http_port']}")

    # ── App Audio-In WS (8776) ─────────────────────────────────────────
    app_audio_srv = await websockets.serve(
        app_audio_in_handler, CONFIG["host"], CONFIG["app_audio_in_port"],
        max_size=50 * 1024 * 1024, ping_interval=30, ping_timeout=60,
    )
    log.info(f"App-AudioIn: ws://localhost:{CONFIG['app_audio_in_port']}")

    # ── App Text-WS (8775+1 = 8778 für Text-Updates an WebView) ───────
    # NEIN: Wir nutzen einen separaten WS-Server für Text
    # Statt 8778 → verwenden wir 8775 + separaten WS auf gleichem Port?
    # Einfacher: Text-WS auf Port app_http_port + 1 = 8778
    app_text_ws_port = CONFIG["app_http_port"] + 3  # 8778
    app_text_srv = await websockets.serve(
        app_text_ws_handler, CONFIG["host"], app_text_ws_port,
        max_size=10 * 1024 * 1024, ping_interval=30, ping_timeout=60,
    )
    log.info(f"App-TextWS:  ws://localhost:{app_text_ws_port}")

    # ── App Audio-Out WS (8777) ────────────────────────────────────────
    app_audio_out_srv = await websockets.serve(
        app_audio_out_handler, CONFIG["host"], CONFIG["app_audio_out_port"],
        max_size=50 * 1024 * 1024, ping_interval=30, ping_timeout=60,
    )
    log.info(f"App-AudioOut: ws://localhost:{CONFIG['app_audio_out_port']}")

    log.info("")
    log.info(f"  Wakeword-Modell: {CONFIG['wake_model_path']}")
    log.info(f"  Command-Modell:  whisper-{CONFIG['cmd_model']}")
    log.info(f"  TTS:             XTTS-API-Server {CONFIG['tts_api_url']}")
    log.info(f"  Wetter:          {CONFIG['weather_city']} ({CONFIG['weather_lat']}, {CONFIG['weather_lon']})")
    log.info(f"  API-URL:         {CONFIG['api_url']}")
    log.info("")
    log.info("  ┌─────────────────────────────────────────────────┐")
    log.info("  │  BROWSER:  8765 (HTTP) / 8766 (WS)             │")
    log.info(f"  │  APP:      {CONFIG['app_http_port']} (HTTP) / {CONFIG['app_audio_in_port']} (Mic-In)         │")
    log.info(f"  │           {CONFIG['app_audio_out_port']} (Audio-Out) / {app_text_ws_port} (Text-WS)     │")
    log.info("  └─────────────────────────────────────────────────┘")
    if dg_config:
        log.info("")
        log.info("  Device Groups:")
        for g in dg_config:
            log.info(f"    '{g.get('name', '?')}': {', '.join(g.get('ips', []))}")
    log.info("")
    log.info("  Sophie ist bereit!  Sage 'Sophie' ...")

    try:
        await asyncio.Future()
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Beende ...")
        timer_manager.stoppe_alle()
        srv.close()
        await srv.wait_closed()
        app_audio_srv.close()
        await app_audio_srv.wait_closed()
        app_text_srv.close()
        await app_text_srv.wait_closed()
        app_audio_out_srv.close()
        await app_audio_out_srv.wait_closed()
        await runner.cleanup()
        await app_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
