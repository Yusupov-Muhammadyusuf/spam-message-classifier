from imap_client import connect_email, fetch_emails, read_email, parse_email
from ml.predict import load_model, load_tokenizer, predict

def main():
    imap = connect_email()

    email_ids = fetch_emails(imap, limit=5)

    tokenizer = load_tokenizer()

    vocab_size = len(tokenizer.word2idx) + 1
    model = load_model(vocab_size=vocab_size)

    for id in email_ids:
        msg = read_email(imap, id)
        subject, body = parse_email(msg)

        text = (subject or "") + " " + (body or "")

        result = predict(text, model, tokenizer)

        print("Subject:", subject)
        print("Pred:", result)
        print("=" * 50)

    imap.logout()

if __name__ == "__main__":
    main()