

export default async function predictFromModel(data: string) {
    const serverURL = process.env.NEXT_PUBLIC_SERVER_URL || 'http://127.0.0.1:8000/predict';

    const response = await fetch(serverURL, {
        method: 'POST',
        headers: {
            'Content-Type': 'text/csv',
            'Accept': 'application/json',
        },
        body: data,
    });
    if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
    }

    const result = await response.json();
    return result; // Assuming the server returns JSON data
}