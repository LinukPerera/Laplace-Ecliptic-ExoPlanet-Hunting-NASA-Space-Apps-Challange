

export default function BackgroundAnimation() {
    const bgStyles = {
        background: 'url("/images/52515250436_4618b5216e_k.jpg") center center / cover no-repeat fixed',
        opacity: 0.5,
    }
    return (
        <div className="background-animation absolute inset-0 z-10" style={bgStyles}>

        </div>
    );
}