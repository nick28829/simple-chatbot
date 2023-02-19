import './App.css';

import { useEffect, useState } from 'react'

function App() {
  const [formData, setFormData] = useState({message: ''});
  const [context, setContext] = useState({});
  const [messages, setMessages] = useState([]);

  function handleFormChange(event) {
    const {name, value} = event.target;
    setFormData(prevFormData => ({
        ...prevFormData,
        [name]: value
    }));
  }

  function handleSubmit(event) {
    event.preventDefault();
    fetch(
      '/api', 
      {
        method: 'POST',
        body: JSON.stringify(
          {
            message: formData.message,
            context
          }
        ),
        headers: {}
      }
    ).then(response => response.json()
    ).then(data => {
      console.log(data)
      if (data.context) setContext(data.context);
      if (data.message) setMessages(prevMessages => [
          ...prevMessages,
          {
            sent: true,
            text: formData.message
          },
          {
            sent: false,
            text: data.message
          }
        ]
      )
      setFormData({message: ''});
    })
  }

  useEffect(() => {
    fetch(
      '/api', 
      {
        method: 'POST',
        body: JSON.stringify(
          {
            message: '',
            context
          }
        ),
        headers: {}
      }
    ).then(response => response.json()
    ).then(data => {
      if (data.context) setContext(data.context);
      if (data.message) setMessages(prevMessages => [
          ...prevMessages,
          {
            sent: false,
            text: data.message
          }
        ]
      )
  })
}, [])

  const messageElements = messages.map(message => {
    return (
      <div className={message.sent ? 'message message-sent' : 'message message-received'}
        dangerouslySetInnerHTML={{ __html: `<p>${message.text}</p>`}
      }>
      </div>
    )
  })

  return (
    <div className='base-container'>
      <header>
       <h1>CV-Bot</h1>
      </header>
      <body>
        <div className='chat'>
          <div className='chat-messages'>
           {messageElements}
          </div>
          <form onSubmit={handleSubmit} className='message-input'>
            <input type='text' className='chat-field' name='message' value={formData.message} onChange={handleFormChange}></input>
            <button className='chat-send-button'>Send</button>
          </form>
        </div>
          <div className='spacing'/>
          <h2>
            About this project:
          </h2>
          <p>
            This project is the result of a rainy Sunday. It is a very simple chatbot. The base is a YAML-file, where all intents
            are listed. The correct intent is found by comparing the w2v-representation of the input with some example sentences for each intent.
            If an intent needs arguments, e.g. a company name, the YAML-file provides a list of possible values. The bot determines the 
            correct value by using the Levenshtein-distance. If the argument is not found, follow-up questions for missing arguments are also
            provided in the configuration.
          </p>
          <h2>
            About Me:
          </h2>
          <p>
            I am Nicklas Bocksberger, a Data Scientist based in Munich. Currently, I'm employed at the Principia Mentis GmbH. Regarding my academic
            career I'm doing my Bachelor's at the University of Applied Sciences Munich with the major Data Science &amp; Scientific Computing.
            I'm interested in Natural Language Processing and I'm hoping to be able to further specialize on this topic in the future. If you are
            interested in what I do you might consider following me on <a href='https://medium.com/@nicklas_bocksberger'>Medium</a> or send me a
            connection request on <a href='https://www.linkedin.com/in/nicklas-bocksberger-8b23b8154/'>LinkedIn</a>.
          </p>
          <div className='spacing'/>
      </body>
      <footer>
        <div className='react-link'>
          built using <a href='https://reactjs.org/'>React</a>
        </div>
        <div className='footer-links'>
          <ul>
            <li>
              <a href='https://github.com/nick28829/simple-chatbot'>GitHub Repository</a>
            </li>
            <li>
              <a href='https://nicklas-bocksberger.com/cv'>My CV as a PDF</a>
            </li>
            <li>
              <a href='https://nicklas-bocksberger.com'>My Website</a>
            </li>
          </ul>
        </div>
      </footer>
    </div>
  );
}

export default App;
